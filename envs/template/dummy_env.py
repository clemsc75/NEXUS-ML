import numpy as np
from typing import Tuple, Dict, Any
from core_engine.interfaces.base_env import IGameEnvironment, EnvSpecs

class DummyEnv(IGameEnvironment):
    """
    Environnement factice pour tester le pipeline d'entraînement
    sans dépendre d'une logique de jeu complexe.
    
    Le but est simple : l'agent doit toujours choisir l'action 0.
    """

    def __init__(self) -> None:
        # État : un vecteur de 4 nombres aléatoires
        self.state_dim = (4,)
        # Actions : 2 choix possibles (0 ou 1)
        self.n_actions = 2
        self.current_step = 0
        self.max_steps = 20  # L'épisode dure 20 pas max

    def reset(self) -> np.ndarray:
        """Reset l'environnement et retourne un état aléatoire."""
        self.current_step = 0
        # Génère un état aléatoire type float32
        return np.random.rand(*self.state_dim).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Simule un pas.
        Règle : Si action == 0 -> Reward +1, Sinon Reward -1.
        """
        self.current_step += 1
        
        # 1. Calcul du nouvel état (bruit aléatoire)
        next_state = np.random.rand(*self.state_dim).astype(np.float32)
        
        # 2. Calcul de la récompense (Logique simple pour apprentissage rapide)
        reward = 1.0 if action == 0 else -1.0
        
        # 3. Vérification de fin d'épisode
        done = self.current_step >= self.max_steps
        
        # 4. Info additionnelle
        info = {"step": self.current_step}
        
        return next_state, reward, done, info

    def get_specs(self) -> EnvSpecs:
        """Retourne les dimensions pour l'agent."""
        return {
            "input_shape": self.state_dim,
            "action_space": self.n_actions
        }

    def render(self) -> None:
        print(f"DummyEnv [Step {self.current_step}]")