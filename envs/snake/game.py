import numpy as np
import random
from typing import Tuple, Dict, Any
from enum import Enum

from core_engine.interfaces.base_env import IGameEnvironment, EnvSpecs

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class SnakeGame(IGameEnvironment):
    """
    Implémentation optimisée du jeu Snake pour l'apprentissage par renforcement.
    L'état est retourné sous forme de grille ou de vecteur de perception.
    """

    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.reset()

    def reset(self) -> np.ndarray:
        # Init Snake au centre
        self.head = [self.grid_size // 2, self.grid_size // 2]
        self.snake = [self.head[:], [self.head[0], self.head[1]-1], [self.head[0], self.head[1]-2]]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 2 # Limite pour éviter les boucles infinies
        
        self._place_food()
        return self._get_state()

    def _place_food(self):
        while True:
            self.food = [random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)]
            if self.food not in self.snake:
                break

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Action: 0=Continue, 1=Turn Right, 2=Turn Left (Relatif à la direction actuelle)
        Cette approche relative facilite l'apprentissage.
        """
        self.steps += 1
        
        # Mise à jour de la direction (Logique relative)
        # Clockwise: [UP, RIGHT, DOWN, LEFT]
        clock_wise = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock_wise.index(self.direction)
        
        if action == 1: # Turn Right
            new_dir = clock_wise[(idx + 1) % 4]
            self.direction = new_dir
        elif action == 2: # Turn Left
            new_dir = clock_wise[(idx - 1) % 4]
            self.direction = new_dir
        # action == 0 -> Pas de changement

        # Mouvement
        x, y = self.head
        if self.direction == Direction.UP: y -= 1
        elif self.direction == Direction.DOWN: y += 1
        elif self.direction == Direction.LEFT: x -= 1
        elif self.direction == Direction.RIGHT: x += 1
        
        self.head = [x, y]
        
        # Récompense et Game Over
        reward = 0
        done = False
        
        # Collision Mur ou Soi-même
        if (x < 0 or x >= self.grid_size or 
            y < 0 or y >= self.grid_size or 
            self.head in self.snake):
            done = True
            reward = -10
        else:
            self.snake.insert(0, self.head)
            
            # Manger la pomme
            if self.head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                self.snake.pop() # Enlever la queue
                # Petite pénalité pour encourager la rapidité (optionnel)
                reward = -0.01

        # Limite de temps
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {"score": self.score}

    def _get_state(self) -> np.ndarray:
        """
        Retourne un état simplifié de 11 valeurs (Booléens convertis en int).
        C'est une représentation classique très efficace pour DQN.
        [Danger Straight, Danger Right, Danger Left, 
         Dir Left, Dir Right, Dir Up, Dir Down,
         Food Left, Food Right, Food Up, Food Down]
        """
        head = self.snake[0]
        
        # Points autour de la tête
        point_l = [head[0] - 1, head[1]]
        point_r = [head[0] + 1, head[1]]
        point_u = [head[0], head[1] - 1]
        point_d = [head[0], head[1] + 1]
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            # Danger Straight
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger Right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger Left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # Move Direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food Location
            self.food[0] < head[0],  # Food Left
            self.food[0] > head[0],  # Food Right
            self.food[1] < head[1],  # Food Up
            self.food[1] > head[1]   # Food Down
        ]
        
        return np.array(state, dtype=int)

    def _is_collision(self, pt) -> bool:
        return (pt[0] < 0 or pt[0] >= self.grid_size or 
                pt[1] < 0 or pt[1] >= self.grid_size or 
                pt in self.snake)

    def get_specs(self) -> EnvSpecs:
        return {
            "input_shape": (11,),
            "action_space": 3 # [Straight, Right, Left]
        }

    def render(self) -> None:
        # On implémentera une vue console simple plus tard ou Pygame
        pass