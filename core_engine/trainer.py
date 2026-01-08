import time
from typing import Dict, Any, Optional
import numpy as np

from core_engine.interfaces.base_env import IGameEnvironment
from core_engine.interfaces.base_agent import IAgent

class Trainer:
    """
    Orchestre l'entraînement : fait interagir l'Agent et l'Environnement.
    """
    
    def __init__(self, env: IGameEnvironment, agent: IAgent, config: Dict[str, Any]):
        self.env = env
        self.agent = agent
        self.config = config
        
        # Paramètres d'entraînement
        self.max_episodes = config.get("max_episodes", 1000)
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

    def train(self) -> Dict[str, Any]:
        """
        Lance la boucle principale d'entraînement.
        """
        metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": []
        }
        
        epsilon = self.epsilon_start
        start_time = time.time()
        
        print(f"🚀 Démarrage de l'entraînement sur {self.device_info()}...")

        for episode in range(1, self.max_episodes + 1):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                # 1. Sélection action
                action = self.agent.select_action(state, epsilon)
                
                # 2. Exécution dans l'environnement
                next_state, reward, done, _ = self.env.step(action)
                
                # 3. Mémorisation
                self.agent.remember(state, action, reward, next_state, done)
                
                # 4. Apprentissage
                train_info = self.agent.train_step()
                if "loss" in train_info and train_info["loss"] != 0:
                    metrics["losses"].append(train_info["loss"])
                
                state = next_state
                total_reward += reward
                steps += 1

            # Fin de l'épisode
            metrics["episode_rewards"].append(total_reward)
            metrics["episode_lengths"].append(steps)
            
            # Decay epsilon
            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)
            
            # Logs périodiques
            if episode % 10 == 0:
                avg_reward = np.mean(metrics["episode_rewards"][-10:])
                print(f"Episode {episode}/{self.max_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

        total_time = time.time() - start_time
        print(f"✅ Entraînement terminé en {total_time:.2f}s")
        
        return metrics

    def device_info(self) -> str:
        if hasattr(self.agent, 'device'):
            return str(self.agent.device).upper()
        return "Unknown Device"