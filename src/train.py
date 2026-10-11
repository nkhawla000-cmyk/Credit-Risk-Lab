"""Baseline documentée : validation séparée, coûts et calibration."""
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import load_data, MODEL_FEATURES, MODEL_NUMERIC, MODEL_CATEGORICAL

SEED = 42
COST_MISSED_DEFAULT = 5.0   # faux negatif : mauvais classe bon
COST_FALSE_ALARM = 1.0      # faux positif : bon classe mauvais


def make_preprocessing():
    return ColumnTransformer(transformers=[
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), MODEL_NUMERIC),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore'))]), MODEL_CATEGORICAL)
    ])


def classification_cost(y_true, probability, threshold):
    pred = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    return COST_MISSED_DEFAULT * int(fn) + COST_FALSE_ALARM * int(fp)


def choose_threshold(y_true, probability):
    # Uniquement sur la validation, jamais sur le test.
    grid = np.linspace(0.05, 0.95, 91)
    costs = [classification_cost(y_true, probability, float(t)) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y_true, proba, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true, proba >= threshold, labels=[0,1]).ravel()
    n = len(y_true)
    return {
        'n': int(n),
        'roc_auc': round(float(roc_auc_score(y_true, proba)), 4),
        'pr_auc': round(float(average_precision_score(y_true, proba)), 4),
        'brier': round(float(brier_score_loss(y_true, proba)), 4),
        'erreur_cout_total': round(float(COST_MISSED_DEFAULT*fn+COST_FALSE_ALARM*fp),2),
        'erreur_cout_par_dossier': round(float((COST_MISSED_DEFAULT*fn+COST_FALSE_ALARM*fp)/n),4),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def run(data_path='data/german.data', root='.'):
    root = Path(root)
    (root/'models').mkdir(parents=True,exist_ok=True)
    (root/'reports').mkdir(parents=True,exist_ok=True)
    df = load_data(data_path)
    # 60/20/20 avec stratification et sous-ensembles mutuellement exclusifs.
    indexes = np.arange(len(df))
    train_idx, temp_idx = train_test_split(indexes, test_size=0.40, random_state=SEED,
                                           stratify=df['default'])
    valid_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SEED,
                                           stratify=df.iloc[temp_idx]['default'])
    X_train, y_train = df.iloc[train_idx][MODEL_FEATURES], df.iloc[train_idx]['default']
    X_valid, y_valid = df.iloc[valid_idx][MODEL_FEATURES], df.iloc[valid_idx]['default']
    X_test, y_test = df.iloc[test_idx][MODEL_FEATURES], df.iloc[test_idx]['default']

    learners = {
        'Regression_logistique': LogisticRegression(max_iter=2000, random_state=SEED),
        'Random_forest': RandomForestClassifier(n_estimators=150, min_samples_leaf=4,
                                                random_state=SEED, n_jobs=-1)
    }
    trials = {}
    champions = {}
    for name, learner in learners.items():
        base = Pipeline([('preprocessing', make_preprocessing()), ('model',learner)])
        calibrated = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        calibrated.fit(X_train, y_train)
        p_valid = calibrated.predict_proba(X_valid)[:,1]
        th = choose_threshold(y_valid, p_valid)
        trials[name] = {'seuil_validation':round(th,3),
                        'validation':evaluate(y_valid, p_valid, th)}
        champions[name] = calibrated

    # Choisir le candidat sur la validation seulement.
    winner = min(trials, key=lambda k:(trials[k]['validation']['erreur_cout_par_dossier'],
                                        trials[k]['validation']['brier']))
    threshold = trials[winner]['seuil_validation']
    p_test = champions[winner].predict_proba(X_test)[:,1]
    report = {
        'dataset':'UCI Statlog German Credit (1994), 1000 exemples historiques',
        'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'random_state':SEED, 'n_train':len(train_idx),'n_validation':len(valid_idx),
        'n_test':len(test_idx), 'cout_fn':COST_MISSED_DEFAULT, 'cout_fp':COST_FALSE_ALARM,
        'sensibles_exclus':['personal_status_sex','foreign_worker','age_years'],
        'candidats_validation':trials, 'modele_retenu':winner,
        'seuil_retenu_sur_validation':threshold,
        'test_final':evaluate(y_test, p_test, threshold),
        'statut':'prototype pedagogique, non utilisable pour decisions bancaires'
    }
    joblib.dump({'model':champions[winner], 'threshold':threshold, 'features':MODEL_FEATURES},
                root/'models/credit_risk_model.joblib')
    (root/'reports/metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    pd.DataFrame({'true_default':y_test.to_numpy(), 'predicted_probability':p_test,
                  'predicted_default':(p_test>=threshold).astype(int)}).to_csv(
                      root/'reports/test_predictions.csv', index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report

if __name__ == '__main__':
    run()
