# NEXUS-ML : Plateforme de Développement et Déploiement ML

## 🚀 Objectif

NEXUS-ML est une plateforme full-stack modulaire conçue pour simplifier le cycle de vie du Machine Learning : de l'expérimentation à la production. Elle met l'accent sur une architecture propre, des standards de développement élevés et une flexibilité maximale.

## 🛠️ Technologies Clés

*   **Backend :** Python 3.10+, FastAPI (Asynchrone)
*   **ML Core :** PyTorch, NumPy
*   **Database :** SQLite (via SQLAlchemy ORM)
*   **Frontend :** Vanilla JavaScript, HTML5, CSS3
*   **Infrastructure :** Docker (à venir)

## 📜 Cahier des Charges

Ce projet suit un cahier des charges technique détaillé disponible à [lien vers le document, si hébergé en ligne] ou consultable dans les documents du projet.

## 🧭 Architecture

L'application est structurée autour de plusieurs modules indépendants :

*   **Backend API :** Gère les requêtes, l'authentification et la logique métier.
*   **Core Engine :** Moteur ML indépendant contenant les agents et les interfaces.
*   **Environnements :** Plugins pour les différents jeux ou environnements RL.
*   **Frontend :** Interface utilisateur web interactive.

## 🏁 Démarrage Rapide

```bash
# Cloner le dépôt
git clone <VOTRE_URL_GITHUB>
cd NEXUS-ML

# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate # Ou .\venv\Scripts\activate sur Windows

# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur backend (développement)
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Accéder à l'API docs à http://localhost:8000/docs