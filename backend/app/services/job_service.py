import uuid
import threading
import asyncio
from typing import Dict
from backend.app.schemas.config import TrainingConfig

# Imports du Core Engine
from envs.snake.game import SnakeGame
from core_engine.agents.dqn_agent import DQNAgent
from core_engine.trainer import Trainer

# Stockage en mémoire simple pour l'instant (sera remplacé par la DB plus tard)
# Structure: { "job_id": { "status": "running", "metrics": {...} } }
active_jobs: Dict[str, Dict] = {}

def run_training_task(job_id: str, config: TrainingConfig):
    """
    Fonction exécutée dans un thread séparé.
    """
    try:
        print(f"Job {job_id}: Started")
        active_jobs[job_id]["status"] = "running"
        
        # 1. Instanciation (Factory Pattern simplifié)
        if config.env_id == "snake_v1":
            env = SnakeGame()
        else:
            raise ValueError(f"Unknown env_id: {config.env_id}")
            
        specs = env.get_specs()
        
        # 2. Agent
        if config.algorithm == "DQN":
            agent = DQNAgent(
                state_dim=specs["input_shape"],
                action_dim=specs["action_space"],
                config=config.hyperparameters
            )
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
            
        # 3. Trainer
        # On injecte les paramètres depuis la config
        trainer_config = config.hyperparameters.copy()
        trainer_config["max_episodes"] = config.max_episodes
        
        trainer = Trainer(env, agent, trainer_config)
        
        # 4. Lancement (Bloquant pour le thread, mais pas pour l'API)
        metrics = trainer.train()
        
        # 5. Sauvegarde résultats
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["result"] = metrics
        print(f"Job {job_id}: Completed")
        
    except Exception as e:
        print(f"Job {job_id}: Failed - {str(e)}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)

async def start_training_job(config: TrainingConfig) -> str:
    """
    Prépare et lance le job en arrière-plan.
    """
    job_id = str(uuid.uuid4())
    
    # Initialisation du statut
    active_jobs[job_id] = {
        "config": config.model_dump(),
        "status": "pending",
        "created_at": "now" # TODO: Timestamp réel
    }
    
    # Lancement dans un thread pour ne pas bloquer l'Event Loop asyncio de FastAPI
    # Note: Pour une vraie prod lourde, on utiliserait Celery, mais Threading suffit ici.
    thread = threading.Thread(target=run_training_task, args=(job_id, config))
    thread.start()
    
    return job_id