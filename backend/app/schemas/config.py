from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class TrainingConfig(BaseModel):
    """
    Le contrat JSON envoyé par le Frontend pour configurer un entraînement.
    """
    experiment_name: str = Field(..., min_length=3, example="Snake_DQN_Alpha")
    env_id: str = Field(..., example="snake_v1")
    algorithm: str = Field(..., example="DQN")
    
    # Hyperparamètres flexibles
    hyperparameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "batch_size": 64,
            "buffer_size": 10000,
            "epsilon_start": 1.0,
            "epsilon_decay": 0.995
        }
    )
    
    max_episodes: int = Field(1000, ge=10, le=100000)
    headless: bool = True

class JobResponse(BaseModel):
    """Réponse immédiate après lancement d'un job."""
    job_id: str
    status: str
    message: str