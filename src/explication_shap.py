"""Contributions SHAP du modèle calibré (sur la validation, pas le test).

Lancer depuis la racine : python -m src.explication_shap
Le calcul peut être long sur un Mac portable. Option : --n-explain 15.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.data import MODEL_CATEGORICAL, MODEL_NUMERIC, load_data
from src.train import SEED

ROOT = Path(__file__).resolve().parents[1]


def main(n_explain=30, n_background=20):
    model_file = ROOT / "models" / "credit_risk_model.joblib"
    if not model_file.exists():
        raise FileNotFoundError("Modèle manquant : exécuter `python -m src.train`.")
    # joblib/pickle : ne charger que vos propres fichiers, jamais un fichier non fiable.
    bundle = joblib.load(model_file)
    calibrated = bundle["model"]
    columns = bundle["features"]
    frame = load_data(ROOT / "data" / "german.data")
    all_indices = np.arange(len(frame))
    train, other = train_test_split(
        all_indices, test_size=0.40, random_state=SEED, stratify=frame["default"]
    )
    validation, _ = train_test_split(
        other, test_size=0.50, random_state=SEED,
        stratify=frame.iloc[other]["default"],
    )

    # SHAP permutation opère sur un tableau numérique ; nous conservons une
    # correspondance réversible entre codes temporaires et catégories d'origine.
    # Les dictionnaires sont appris uniquement sur train + validation (sans test).
    working = frame[columns].copy()
    decoding = {}
    for col in MODEL_CATEGORICAL:
        observed = pd.concat([working.iloc[train][col], working.iloc[validation][col]])
        categories = sorted(observed.astype(str).unique())
        decoding[col] = dict(enumerate(categories))
        encoding = {value: code for code, value in decoding[col].items()}
        working[col] = working[col].map(encoding)

    background = working.iloc[train][columns].sample(
        n=min(n_background, len(train)), random_state=SEED
    ).reset_index(drop=True)
    # Même échantillon déterministe entre deux exécutions.
    examples_indices = frame.iloc[validation].sample(
        n=min(n_explain, len(validation)), random_state=SEED
    ).index
    # Les indices df d'origine sont 0..n-1 pour German Credit.
    examples = working.loc[examples_indices, columns].reset_index(drop=True)
    original_display = frame.loc[examples_indices, columns].reset_index(drop=True)

    def predict_default(values):
        x = pd.DataFrame(values, columns=columns)
        for col in MODEL_NUMERIC:
            x[col] = pd.to_numeric(x[col], errors="raise")
        for col in MODEL_CATEGORICAL:
            code = pd.to_numeric(x[col], errors="raise").round().astype(int)
            x[col] = code.map(decoding[col])
            if x[col].isna().any():
                raise ValueError(f"Catégorie non reconnue pendant SHAP : {col}")
        return calibrated.predict_proba(x)[:, 1]

    print(f"SHAP permutation pour {len(examples)} dossiers de validation...")
    explainer = shap.Explainer(
        predict_default, shap.maskers.Independent(background), algorithm="permutation"
    )
    explanation = explainer(examples, max_evals=2 * len(columns) + 1)
    # Afficher les libellés catégoriels réels (A11, A34...), et non les codes
    # temporaires (0, 1, 2...) utilisés pour expliquer le modèle.
    explanation.display_data = original_display.to_numpy(dtype=object)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    plt.figure()
    shap.plots.bar(explanation, max_display=12, show=False)
    plt.savefig(reports / "shap_global.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=12, show=False)
    plt.savefig(reports / "shap_individuel.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    meta = {
        "n_explained": len(examples),
        "model_selected": json.loads((reports / "metrics.json").read_text(encoding="utf-8")).get("modele_retenu", "inconnu") if (reports / "metrics.json").exists() else "inconnu",
        "background_size": len(background),
        "data_split": "validation uniquement pour les cas expliqués ; entraînement pour le fond",
        "method": "SHAP permutation pour prédiction probabiliste du modèle calibré",
        "limitations": "Importance moyennée sur un sous-échantillon, pas sur une population bancaire",
    }
    (reports / "shap_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SHAP terminé : reports/shap_global.png et reports/shap_individuel.png")
    print("Légende : les contributions expliquent le modèle et non la causalité.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-explain", type=int, default=30)
    parser.add_argument("--background", type=int, default=20)
    options = parser.parse_args()
    if options.n_explain < 1 or options.background < 1:
        parser.error("n-explain et background doivent être positifs")
    main(options.n_explain, options.background)
