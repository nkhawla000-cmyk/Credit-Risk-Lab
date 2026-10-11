"""Dashboard de démonstration : pas une application de décision de crédit."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
METRICS = REPORTS / "metrics.json"
PREDICTIONS = REPORTS / "test_predictions.csv"

st.set_page_config(page_title="Credit Risk Lab | Explainable ML", layout="wide")
st.title("Credit Risk Lab | Scoring explicable")
st.warning(
    "Prototype pédagogique : German Credit (1994), petit échantillon historique. "
    "Ne pas utiliser pour des décisions d'octroi ou de refus de crédit."
)

if not METRICS.exists() or not PREDICTIONS.exists():
    st.info(
        "Résultats locaux absents. Télécharge les données via "
        "`python scripts/download_data.py` (ou manuellement), puis exécute "
        "`python -m src.train` depuis la racine du projet."
    )
    st.stop()

report = json.loads(METRICS.read_text(encoding="utf-8"))
predictions = pd.read_csv(PREDICTIONS)
test = report["test_final"]
winner = report["modele_retenu"]
threshold = float(report["seuil_retenu_sur_validation"])

st.header("Évaluation sur l'échantillon de test indépendant")
a, b, c, d = st.columns(4)
a.metric("ROC-AUC", f"{test['roc_auc']:.3f}")
b.metric("PR-AUC", f"{test['pr_auc']:.3f}")
c.metric("Score de Brier", f"{test['brier']:.3f}")
d.metric("Coût / dossier (unités)", f"{test['erreur_cout_par_dossier']:.3f}")
st.write(f"**Modèle sélectionné sur validation :** {winner}  |  "
         f"**Seuil fixé sur validation :** {threshold:.2f}")
st.caption(
    "Coûts illustratifs UCI : faux négatif = 5, faux positif = 1. "
    "Le modèle et son seuil ne sont pas réajustés avec le test."
)

left, right = st.columns(2)
with left:
    st.subheader("Distribution des probabilités prédites")
    hist = pd.cut(predictions["predicted_probability"], bins=10).value_counts(sort=False)
    hist.index = hist.index.astype(str)
    st.bar_chart(hist)
with right:
    st.subheader("Matrice de confusion (test)")
    matrix = pd.DataFrame(
        {"Prévision bon risque": [test["tn"], test["fn"]],
         "Prévision mauvais risque": [test["fp"], test["tp"]]},
        index=["Vrai bon risque", "Vrai mauvais risque"],
    )
    st.dataframe(matrix, use_container_width=True)

st.subheader("Comparaison sur validation (pas sur le test)")
st.dataframe(
    pd.DataFrame([
        {"modele": name, "seuil": candidate["seuil_validation"], **candidate["validation"]}
        for name, candidate in report["candidats_validation"].items()
    ]),
    use_container_width=True,
)

st.divider()
st.header("Explicabilité : contributions SHAP")
st.write(
    "Les valeurs SHAP décrivent les contributions aux probabilités prédites "
    "par le modèle calibré, et non les causes du risque."
)
shap_metadata_path = REPORTS / "shap_metadata.json"
if shap_metadata_path.exists():
    shap_metadata = json.loads(shap_metadata_path.read_text(encoding="utf-8"))
    n_explained = shap_metadata.get("n_explained", "inconnu")
    st.caption(f"Importance calculée sur {n_explained} dossiers de validation échantillonnés.")
else:
    st.caption(
        "Ancienne analyse illustrative sur cinq dossiers de validation. "
        "Pour une étude plus étendue : `python -m src.explication_shap`."
    )
if shap_metadata_path.exists():
    explained_model = shap_metadata.get("model_selected")
    if explained_model and explained_model != winner:
        st.warning("Les graphiques SHAP ont été générés pour un autre modèle. Recalcule-les.")

for title, filename in [
    ("Importance moyenne absolue des variables", "shap_global.png"),
    ("Explication d'un dossier de validation", "shap_individuel.png"),
]:
    path = REPORTS / filename
    st.subheader(title)
    if path.exists():
        st.image(str(path), width="stretch")
    else:
        st.info("Graphique absent. Lance `python -m src.explication_shap`.")

st.divider()
st.header("Validation statistique descriptive")
advanced_path = REPORTS / "validation_avancee.json"
if advanced_path.exists():
    advanced = json.loads(advanced_path.read_text(encoding="utf-8"))
    bounds = advanced.get("roc_auc_ic95_bootstrap_stratifie")
    if bounds and len(bounds) == 2:
        st.info(
            f"ROC-AUC = {advanced['roc_auc_test']:.3f} | "
            f"IC bootstrap stratifié 95 % : [{bounds[0]:.3f} ; {bounds[1]:.3f}] "
            "(estimation sur 200 observations)."
        )
else:
    st.info("Calculs avancés absents. Lance `python -m src.validation_avancee`.")

for title, filename in [
    ("Courbe ROC", "roc_test.png"),
    ("Calibration des probabilités", "calibration_test.png"),
]:
    path = REPORTS / filename
    st.subheader(title)
    if path.exists():
        st.image(str(path), width="stretch")
    else:
        st.info("Graphique absent : lance `python -m src.validation_avancee`.")

st.caption(
    "Limites : données anciennes et non représentatives des demandes actuelles, "
    "test réduit, risque de sélection sur validation, analyses exploratoires "
    "aussi menées sur les 1 000 observations. Aucun usage bancaire réel."
)
