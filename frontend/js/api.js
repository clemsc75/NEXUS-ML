const API_BASE_URL = "http://127.0.0.1:8000";

/**
 * Vérifie si le backend est en ligne.
 */
export async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        return response.ok;
    } catch (error) {
        console.error("API offline", error);
        return false;
    }
}

/**
 * Envoie la configuration pour démarrer un job.
 */
export async function startJob(config) {
    const response = await fetch(`${API_BASE_URL}/jobs/start`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(config)
    });

    if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
    }
    return await response.json();
}

/**
 * Récupère le statut et les résultats d'un job.
 */
export async function getJobStatus(jobId) {
    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
    if (!response.ok) {
        throw new Error("Impossible de récupérer le statut du job");
    }
    return await response.json();
}