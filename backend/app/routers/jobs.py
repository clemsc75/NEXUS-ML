from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.app.schemas.config import TrainingConfig, JobResponse
from backend.app.services import job_service

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"]
)

@router.post("/start", response_model=JobResponse)
async def start_job(config: TrainingConfig):
    """
    Lance une nouvelle session d'entraînement.
    """
    try:
        job_id = await job_service.start_training_job(config)
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Training job initialized successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """
    Récupère le statut et les résultats d'un job.
    """
    job = job_service.active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job