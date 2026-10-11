"""Lecture des données German Credit (UCI, ID 144)."""
from pathlib import Path
import pandas as pd

FEATURES = [
    'checking_status','duration_months','credit_history','purpose','credit_amount',
    'savings_status','employment_length','installment_rate','personal_status_sex',
    'other_debtors','residence_years','property_type','age_years','other_installment_plans',
    'housing','existing_credits','job','dependents','telephone','foreign_worker'
]
NUMERIC = [
    'duration_months','credit_amount','installment_rate','residence_years',
    'age_years','existing_credits','dependents'
]
# Variables sensibles non utilisees pour entrainer : volontaire et conservateur.
# Attention : les autres variables peuvent aussi contenir des proxys.
SENSITIVE = ['personal_status_sex','foreign_worker','age_years']
CATEGORICAL = [f for f in FEATURES if f not in NUMERIC]
MODEL_FEATURES = [f for f in FEATURES if f not in SENSITIVE]
MODEL_NUMERIC = [f for f in MODEL_FEATURES if f in NUMERIC]
MODEL_CATEGORICAL = [f for f in MODEL_FEATURES if f in CATEGORICAL]

def load_data(path='data/german.data'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. Executer 'python scripts/download_data.py' "
            "ou deposer le fichier german.data de l'UCI dans data/."
        )
    df = pd.read_csv(path, sep=r'\s+', header=None, names=FEATURES+['label'], engine='python')
    if df.shape[1] != 21 or df.isna().all(axis=1).any():
        raise ValueError('Structure inattendue pour le fichier German Credit.')
    if not set(df.label.unique()).issubset({1,2}):
        raise ValueError('Les classes attendues sont 1 (bon) et 2 (mauvais).')
    df['default'] = (df['label'] == 2).astype(int)
    df = df.drop(columns='label')
    for name in NUMERIC:
        df[name] = pd.to_numeric(df[name], errors='raise')
    for name in CATEGORICAL:
        df[name] = df[name].astype(str)
    return df
