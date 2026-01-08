import { checkHealth, startJob, getJobStatus } from './api.js';
import { initChart, updateChartData } from './charts.js';

// --- DOM Elements ---
const serverStatus = document.getElementById('serverStatus');
const form = document.getElementById('trainingForm');
const startBtn = document.getElementById('startBtn');
const jobInfo = document.getElementById('jobInfo');
const currentJobId = document.getElementById('currentJobId');
const jobStatusSpan = document.getElementById('jobStatus');

let pollInterval = null;

// --- Initialization ---
async function init() {
    initChart();
    
    // Vérification de la santé du serveur
    const isOnline = await checkHealth();
    if (isOnline) {
        serverStatus.textContent = "Connecté 🟢";
        serverStatus.style.color = "#22c55e";
        startBtn.disabled = false;
    } else {
        serverStatus.textContent = "Hors Ligne 🔴";
        serverStatus.style.color = "#ef4444";
        startBtn.disabled = true;
    }
}

// --- Event Listeners ---
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // 1. Récupération des données du formulaire
    const config = {
        experiment_name: document.getElementById('expName').value,
        env_id: document.getElementById('envId').value,
        algorithm: "DQN",
        max_episodes: parseInt(document.getElementById('episodes').value),
        hyperparameters: {
            learning_rate: 0.001, // Valeurs par défaut pour l'instant
            gamma: 0.99,
            batch_size: 64
        }
    };

    try {
        // 2. Désactiver le bouton pour éviter le double-clic
        startBtn.disabled = true;
        startBtn.textContent = "Initialisation...";

        // 3. Appel API
        const data = await startJob(config);
        
        // 4. Mise à jour UI
        jobInfo.classList.remove('hidden');
        currentJobId.textContent = data.job_id;
        jobStatusSpan.textContent = "Démarré";
        
        // 5. Lancer le polling (surveillance)
        startPolling(data.job_id);

    } catch (error) {
        alert("Erreur: " + error.message);
        startBtn.disabled = false;
        startBtn.textContent = "Lancer l'entraînement 🚀";
    }
});

// --- Polling Logic ---
function startPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const jobData = await getJobStatus(jobId);
            jobStatusSpan.textContent = jobData.status.toUpperCase();

            // Si le job est terminé
            if (jobData.status === 'completed') {
                clearInterval(pollInterval);
                startBtn.disabled = false;
                startBtn.textContent = "Lancer l'entraînement 🚀";
                jobStatusSpan.style.color = "#22c55e";

                // Afficher les résultats sur le graph
                if (jobData.result && jobData.result.episode_rewards) {
                    updateChartData(jobData.result.episode_rewards);
                }
            } 
            // Si le job a échoué
            else if (jobData.status === 'failed') {
                clearInterval(pollInterval);
                startBtn.disabled = false;
                startBtn.textContent = "Réessayer";
                jobStatusSpan.style.color = "#ef4444";
                alert("Erreur d'entraînement: " + jobData.error);
            }

        } catch (error) {
            console.error("Polling error", error);
        }
    }, 1000); // Vérifie toutes les 1 secondes
}

// Lancement au chargement de la page
window.addEventListener('load', init);