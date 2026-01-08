let scoreChart = null;

export function initChart() {
    const ctx = document.getElementById('scoreChart').getContext('2d');
    
    scoreChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], // Numéros d'épisodes
            datasets: [{
                label: 'Score (Récompense totale)',
                data: [],
                borderColor: '#38bdf8', // Cyan
                backgroundColor: 'rgba(56, 189, 248, 0.1)',
                borderWidth: 2,
                tension: 0.3, // Courbe légèrement lissée
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: '#e2e8f0' } }
            },
            scales: {
                y: {
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

/**
 * Met à jour le graphique avec une liste de scores.
 * @param {Array} rewards - Liste des scores [10, 12, -5, ...]
 */
export function updateChartData(rewards) {
    if (!scoreChart) return;

    // Génère les labels [1, 2, 3...]
    const labels = rewards.map((_, index) => index + 1);

    scoreChart.data.labels = labels;
    scoreChart.data.datasets[0].data = rewards;
    scoreChart.update();
}