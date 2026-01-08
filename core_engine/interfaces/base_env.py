from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, TypedDict
import numpy as np

class EnvSpecs(TypedDict):
    """Définition stricte de la structure des spécifications."""
    input_shape: Tuple[int, ...]
    action_space: int

class IGameEnvironment(ABC):
    """
    Interface standard pour tous les environnements de jeu (Pattern Strategy).
    Permet l'interchangeabilité des environnements sans modifier l'agent ML.
    """

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Réinitialise l'environnement à son état initial.

        Returns:
            np.ndarray: L'état initial du jeu sous forme de tenseur/array.
                        La forme (shape) doit correspondre à get_specs()['input_shape'].
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement.

        Args:
            action (int): L'index de l'action à exécuter (doit être < action_space).

        Returns:
            Tuple contenant:
                - next_state (np.ndarray): Le nouvel état après l'action.
                - reward (float): La récompense obtenue (positive ou négative).
                - done (bool): True si l'épisode est terminé (victoire/défaite).
                - info (Dict[str, Any]): Métadonnées (ex: score brut, temps).
        """
        pass

    @abstractmethod
    def get_specs(self) -> EnvSpecs:
        """
        Retourne les métadonnées techniques de l'environnement nécessaires
        pour construire le réseau de neurones.

        Returns:
            EnvSpecs: Dictionnaire typé contenant la forme d'entrée et l'espace d'action.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        Affiche l'état du jeu (console ou fenêtre graphique).
        Optionnel pour l'entraînement headless, mais utile pour le debug.
        """
        pass