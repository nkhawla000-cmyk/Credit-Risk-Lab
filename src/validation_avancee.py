"""Validation descriptive post-selection de Credit Risk Lab.

A executer depuis la racine : python -m src.validation_avancee
Le choix du modele et du seuil ne doit PAS etre modifie avec les resultats de test.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, confusion_matrix

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports"
PREDICTIONS = REPORTS / "test_predictions.csv"
METRICS = REPORTS / "metrics.json"
SEED = 42
N_BOOT = 2000


def run():
    if not PREDICTIONS.is_file() or not METRICS.is_file():
        raise FileNotFoundError("Lancer d'abord : python -m src.train (metrics.json ou test_predictions.csv manque).")
    REPORTS.mkdir(exist_ok=True, parents=True)
    d = pd.read_csv(PREDICTIONS)
    expected = {"true_default", "predicted_probability", "predicted_default"}
    if not expected.issubset(d.columns):
        raise ValueError(f"Colonnes manquantes : {sorted(expected - set(d.columns))}")
    y = d["true_default"].to_numpy(dtype=int)
    p = d["predicted_probability"].to_numpy(dtype=float)
    yh = d["predicted_default"].to_numpy(dtype=int)
    if not np.isfinite(p).all() or not np.all((0 <= p) & (p <= 1)):
        raise ValueError("Probabilites manquantes ou hors de [0,1].")
    if len(np.unique(y)) != 2 or not set(np.unique(y)).issubset({0,1}):
        raise ValueError("La cible doit contenir les deux classes 0 et 1.")
    metadata = json.loads(METRICS.read_text(encoding="utf-8"))
    threshold = float(metadata["seuil_retenu_sur_validation"])
    if not np.array_equal(yh, (p >= threshold).astype(int)):
        raise ValueError("predicted_default ne correspond pas au seuil fige sur validation.")

    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0,1]).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Bootstrap stratifie des 2 classes, sur le test pour mesurer l'incertitude,
    # sans choisir de nouvelles variables, modele ou seuil.
    rng = np.random.default_rng(SEED)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ind = np.r_[rng.choice(idx0, len(idx0), replace=True),
                    rng.choice(idx1, len(idx1), replace=True)]
        draws[i] = roc_auc_score(y[ind], p[ind])
    ci_lo, ci_hi = np.quantile(draws, [0.025, 0.975])

    # Calibration descriptive: jusqu'a 5 groupes de probabilites de taille proche.
    groups = pd.DataFrame({"p":p, "observed":y})
    groups["bin"] = pd.qcut(groups["p"], q=5, duplicates="drop")
    calibration = (groups.groupby("bin", observed=True)
                   .agg(effectif=("observed", "size"),
                        proba_moyenne=("p", "mean"),
                        taux_observe=("observed", "mean"))
                   .reset_index(drop=True))
    calibration.to_csv(REPORTS / "calibration_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot([0,1], [0,1], linestyle="--", linewidth=1.3, label="Calibration parfaite")
    ax.plot(calibration["proba_moyenne"], calibration["taux_observe"],
            marker="o", linewidth=1.8, label="Observations par quintile")
    for _, row in calibration.iterrows():
        ax.annotate(f"n={int(row['effectif'])}",
                    (row["proba_moyenne"], row["taux_observe"]),
                    xytext=(5,5), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Probabilite moyenne predite", ylabel="Frequence observee de mauvais risque",
           title=f"Calibration sur test (n={len(y)})", xlim=(0,1), ylim=(0,1))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration_test.png", dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(fpr, tpr, linewidth=2.0, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0,1], [0,1], linestyle="--", linewidth=1.0, label="Classifieur aleatoire")
    ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs",
           title=f"ROC sur test - IC bootstrap 95 % [{ci_lo:.3f}; {ci_hi:.3f}]",
           xlim=(0,1), ylim=(0,1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS / "roc_test.png", dpi=180)
    plt.close(fig)

    result = {"n_test":int(len(y)), "positifs_reels":int(y.sum()),
              "modele_selectionne_sur_validation":metadata["modele_retenu"],
              "seuil_fixe_sur_validation":threshold,
              "roc_auc_test":round(float(auc),4),
              "roc_auc_ic95_bootstrap_stratifie":[round(float(ci_lo),4),round(float(ci_hi),4)],
              "score_brier":round(float(brier),4),
              "rappel_mauvais_risque":round(float(recall),4),
              "specificite_bon_risque":round(float(specificity),4),
              "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
              "remarque":"Evaluation retrospective d'un prototype pedagogique; ne pas optimiser le modele sur le test."}
    (REPORTS / "validation_avancee.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nVALIDATION AVANCEE (TEST)\n" + json.dumps(result, indent=2, ensure_ascii=False))
    print("\nFichiers crees : reports/calibration_test.png, reports/roc_test.png,")
    print("reports/calibration_table.csv et reports/validation_avancee.json")
    return result


if __name__ == "__main__":
    run()
