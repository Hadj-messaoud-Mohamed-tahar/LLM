# Démonstrateur Chatbot Creaexpertech (Streamlit + Ollama)

Ce projet est une démo séparée (hors production) d’un chatbot “plateforme de formation”.
L’objectif est de reproduire la logique métier du chatbot Creaexpertech sans connecter la vraie plateforme.

## Fonctionnalités

- Chat Streamlit avec assistant “strict” : répond uniquement sur la plateforme (formations, outils, abonnements, accès).
- Hors-sujet → réponse exacte :  
  `Je suis désolé, mais je ne peux répondre qu'aux questions concernant la plateforme Creaexpertech et ses services.`
- Recommandations : 2–4 formations + 1–2 outils (sélection par scoring).
- Catalogue “à jour” : lecture depuis CSV locaux + bouton de rechargement.
- Abonnements : page HTML locale affichée dans Streamlit (onglet Tarifs) + contexte injecté au modèle pour répondre sur les tarifs.
- Fallback sans IA : si Ollama est indisponible / erreur, une réponse basée sur le ranking est renvoyée.
- Pages “Présentation”, “Data audit”, “Évaluation”, “Résultats” pour le rendu (exploration/nettoyage/métriques).

## Arborescence

- `app.py` : application Streamlit (tout-en-un)
- `modules_rows.csv` : catalogue des formations (source)
- `tools_rows.csv` : catalogue des outils (source)
- `..\pricing exemple.html` : page abonnements (source HTML)
- `eval_set.json` : jeu de test (questions + labels) pour métriques
- `requirements.txt` : dépendances Python

## Prérequis

- Windows / macOS / Linux
- Python 3.10+ recommandé
- Ollama installé et lancé en local : http://localhost:11434

Modèles Ollama :
- Un modèle “chat” (ex : `llama3:latest`)
- (Optionnel) un modèle “embeddings” (ex : `nomic-embed-text`) pour la comparaison A/B

## Installation

Dans un terminal :

```bash
cd ollama_streamlit_demo
python -m pip install -r requirements.txt
```

## Lancer l’application

```bash
cd ollama_streamlit_demo
streamlit run app.py
```

Puis ouvrir l’URL affichée par Streamlit.

## Config (dans la sidebar)

- URL Ollama : par défaut `http://localhost:11434`
- Modèle chat : liste récupérée depuis Ollama (`/api/tags`)
- Modèle embeddings : par défaut `nomic-embed-text`  
  Si le modèle n’est pas installé, la comparaison embeddings est ignorée.

Installation du modèle embeddings (si besoin) :

```bash
ollama pull nomic-embed-text
```

## Données (CSV) : exploration + nettoyage

La page **Data audit** affiche :
- Statistiques “avant/après” (lignes, manquants, doublons)
- Exemples “avant/après” (10 lignes)
- Distributions (difficulty, niveau, métier, plan_required, etc.)

Le nettoyage appliqué est volontairement simple :
- trim + normalisation des espaces
- normalisation de certains champs (difficulty/niveau)
- tags → liste nettoyée + dédoublonnée
- durée → conversion int, valeurs invalides → null
- suppression doublons (sur title, insensible à la casse)

## Évaluation & résultats

Le fichier `eval_set.json` contient des questions avec :
- `intent` (greeting, pricing, tools, catalogue, off_topic)
- `expected_refusal` (bool)
- `expected_items` (optionnel) : titres attendus dans le top-k

Dans Streamlit :
- Page **Évaluation** : exécute l’évaluation, affiche les métriques, export CSV.
- Page **Résultats** : graphiques et commentaire synthétique, comparaison token-overlap vs embeddings (si disponible).

## Reproduire / modifier

- Pour ajouter des formations/outils : modifier `modules_rows.csv` et `tools_rows.csv`, puis cliquer sur “Recharger données”.
- Pour modifier les tests : éditer `eval_set.json`.
- Pour les abonnements : mettre à jour `pricing exemple.html` (le chemin est configurable dans la sidebar).

## Dépannage

- Le chat ne répond pas :
  - Vérifier que Ollama tourne : http://localhost:11434
  - Vérifier le modèle chat sélectionné (il doit exister dans Ollama)
- Embeddings non disponibles :
  - Installer le modèle embeddings : `ollama pull nomic-embed-text`

