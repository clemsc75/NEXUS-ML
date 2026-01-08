from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers import jobs

app = FastAPI(
    title="NEXUS-ML API",
    version="1.0.0",
    description="API de pilotage pour plateforme d'entraînement ML"
)

# Configuration CORS (Pour autoriser le Frontend JS)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:5500", # Port par défaut de Live Server VSCode
    "*" # A restreindre en vraie production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(jobs.router)

@app.get("/")
def read_root():
    return {"status": "online", "system": "NEXUS-ML v1.0"}