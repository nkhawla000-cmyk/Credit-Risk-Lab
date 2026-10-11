# Audit du projet Credit Risk Lab — 11 octobre 2026

## Périmètre et preuves

Audit réalisé à partir du ZIP transmis par l'utilisatrice. Code inspecté : `src/`, `scripts/`, `app.py`, `tests/`, `requirements.txt`, README et résultats existants. Les quatre tests initiaux ont été exécutés dans l'environnement d'audit et ont réussi. Le fichier `data/german.data` n'était **pas** dans l'archive (exclusion volontaire), donc aucun entraînement réel n'a été relancé ici. Le modèle `.joblib` a été inspecté comme fichier binaire présent **sans être désérialisé**, par sécurité.

## Résultats numériques présents dans les fichiers

- Échantillon : 1 000 dossiers ; split 600/200/200, stratifié.
- Régression logistique (validation) : ROC-AUC 0,8012 ; coût total 89.
- Random Forest calibré (validation) : ROC-AUC 0,8132 ; coût total 85 ; seuil 0,26.
- Random Forest sur test : ROC-AUC 0,7779 ; PR-AUC 0,6091 ; Brier 0,1676 ; TN 91, FP 49, FN 13, TP 47.
- IC 95 % bootstrap stratifié de l'AUC : [0,7111 ; 0,8426].
- Analyse exploratoire Khi-deux à cinq groupes : chi² = 20,5492, ddl = 4, p ≈ 0,000389 ; association statistique, **pas causalité**.

## Corrections livrées

| Constat | Correction |
|---|---|
| Bibliothèques SHAP, Altair, SciPy, Matplotlib absentes du fichier de dépendances | Ajout dans `requirements.txt` |
| Dashboard avec ROC-AUC et intervalle codés en dur | Lecture dynamique de `metrics.json` et `validation_avancee.json` |
| Texte « SHAP à venir » malgré l'intégration | Texte remplacé par descriptions du travail réellement fait |
| SHAP global sur 5 dossiers, codes catégoriels numériques peu lisibles | Script actualisé : 30 dossiers par défaut (option réglable), libellés de catégories originaux, métadonnées |
| Gitignore insuffisant pour l'environnement et les artefacts locaux | Ajustement du `.gitignore` ; archive publique sans données brutes, modèle sérialisé, prédictions individuelles |
| README obsolète | README anglais avec reproduction, résultats et limites explicites |
| Seulement 4 tests étroits | Ajout de tests pour le format des données et les métriques (voir `tests/`) |

## Risques méthodologiques restant ouverts

- Les graphiques EDA et le test Khi-deux ont été calculés sur les **1 000 lignes**, y compris les lignes désignées ultérieurement comme test final. Aucune extraction de caractéristiques à partir des cibles test dans `train.py` n'a été détectée ; toutefois, l'analyse des hypothèses postérieure à l'accès aux tests doit rester explicitement exploratoire.
- Les deux candidats sont choisis par coût sur 200 validations. Leur différence de coût est faible et ne prouve pas la supériorité générale d'un algorithme.
- L'intervalle de confiance bootstrap traite le modèle comme **fixé** et ne reflète pas l'incertitude liée à l'entraînement, à la calibration, ni à la sélection du modèle.
- Les probabilités ne sont pas totalement calibrées sur chaque sous-groupe ; les courbes par quintile sont bruyantes.
- Les colonnes sensibles ne sont pas utilisées dans le modèle mais des variables proxy peuvent rester ; aucun audit d'équité opérationnelle ou juridique n'a été réalisé.
- Aucune étude externe/longitudinale, aucune API production ou supervision MLOps n'existe pour le moment.

## État des vérifications

- Vérifications textuelles et compilation Python : faites dans cet audit.
- Tests automatisés : exécutés dans l'environnement d'audit (voir résultat dans la réponse).
- Compatibilité de la **nouvelle** application Streamlit et du **nouveau** script SHAP : à valider sur le Mac, les dépendances Streamlit et Altair n'étant pas disponibles dans l'environnement d'audit. Les **anciennes** versions ont été exécutées avec succès sur le Mac par l'utilisatrice.
- Résultats historiques : repris des CSV et JSON contenus dans le ZIP, non recalculés à partir des données brutes absentes.

## Commandes après installation sur le Mac

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train
python -m src.validation_avancee
python -m src.explication_shap --n-explain 30
python -m streamlit run app.py
```

Ne pas déplacer les fichiers dans `__pycache__`. Ne jamais publier des secrets, `data/german.data`, `models/credit_risk_model.joblib`, ni `reports/test_predictions.csv`.

**Hygiène des données :** la nouvelle version du script d’entraînement ne conserve plus `age_years` dans `test_predictions.csv` et enregistre l’empreinte SHA-256 du fichier source.
