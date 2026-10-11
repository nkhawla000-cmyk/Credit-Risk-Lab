# Sprints avant / après le forum ECC

## J1 — Objectif : reproductibilité
- Installer et exécuter le projet ; lancer les tests.
- Vérifier nombre d'observations, types, équilibre des classes, provenance.
- Décrire le coût des faux négatifs/positifs et les limites du jeu UCI.

## J2 — Objectif : statistiques et ML
- Inspecter distributions, valeurs atypiques, relations entre variables.
- Expliquer les coefficients de la régression logistique en odds ratio.
- Comparer ROC-AUC, PR-AUC, calibration et coût par dossier.

## J3 — Objectif : démonstration
- Exécuter le dashboard et vérifier tous les nombres du rapport.
- Préparer un pitch de 90 secondes et un diagramme du pipeline.
- Écrire un README de résultats daté **après** exécution.

## J4 — Objectif : différence concurrentielle
- Ajouter SHAP après validation d'une version compatible.
- Discuter biais de sélection et audit de biais discriminatoires.
- Ajouter un premier test de robustesse bootstrap (sur jeu de test ou nouvelles périodes selon données).

## Ensuite (PFE)
- MLflow + monitoring de dérive + API + Docker.
- Migration vers des données plus récentes et une validation temporelle.

## Questions d'entretien à maîtriser
1. Pourquoi stratifier l'échantillon ?
2. Pourquoi choisir le seuil sur la validation et non sur le test ?
3. Pourquoi la ROC-AUC ne suffit pas à juger un scoring ?
4. Pourquoi calibrer les probabilités ?
5. Comment évaluer l'équité du modèle sans confondre discrimination et performance ?
6. Qu'est-ce qu'un changement de population (data drift) ?
