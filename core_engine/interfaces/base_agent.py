from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class IAgent(ABC):
    """
    Interface abstraite pour tous les agents de Reinforcement Learning.
    Assure que le Trainer peut utiliser n'importe quel algo (DQN, PPO, etc.) de façon interchangeable.
    """

    @abstractmethod
    def select_action(self, state: Any, epsilon: float = 0.0) -> int:
        """
        Choisit une action basée sur l'état actuel.
        
        Args:
            state: L'état actuel de l'environnement.
            epsilon: Taux d'exploration (0.0 = full exploitation, 1.0 = full random).
            
        Returns:
            int: L'index de l'action choisie.
        """
        pass

    @abstractmethod
    def train_step(self) -> Dict[str, float]:
        """
        Effectue une étape d'apprentissage (mise à jour des poids du réseau).
        
        Returns:
            Dict[str, float]: Dictionnaire des métriques (ex: {'loss': 0.025}).
        """
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done) -> None:
        """
        Enregistre une transition dans la mémoire de l'agent (Replay Buffer).
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Sauvegarde les poids du modèle."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Charge les poids du modèle."""
        pass