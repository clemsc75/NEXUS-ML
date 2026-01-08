import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, Tuple, Any
from collections import deque

from core_engine.interfaces.base_agent import IAgent

# --- Utilitaire de détection Hardware ---
def get_best_device() -> torch.device:
    """
    Détecte automatiquement le meilleur matériel disponible.
    Supporte : CUDA (Nvidia), MPS (Mac), XPU (Intel Arc), CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Support Intel Arc (via Intel Extension for PyTorch)
    # Note: nécessite 'pip install intel-extension-for-pytorch' et 'import intel_extension_for_pytorch'
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device("xpu")
    except ImportError:
        pass
        
    if torch.backends.mps.is_available():
        return torch.device("mps")
        
    return torch.device("cpu")

# --- Réseau de Neurones ---
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- Agent DQN ---
class DQNAgent(IAgent):
    """
    Implémentation de l'algorithme Deep Q-Network (DQN).
    """

    def __init__(self, state_dim: Tuple[int, ...], action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim[0] if isinstance(state_dim, tuple) else state_dim
        self.action_dim = action_dim
        self.config = config

        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 64)
        self.lr = config.get("learning_rate", 1e-3)
        
        # 🟢 Utilisation de la détection avancée
        self.device = get_best_device()

        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        buffer_size = config.get("buffer_size", 10000)
        self.memory = deque(maxlen=buffer_size)

    # ... (Le reste des méthodes: select_action, remember, train_step, save, load restent identiques)
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> Dict[str, float]:
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (self.gamma * next_q * (1 - dones_t))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())