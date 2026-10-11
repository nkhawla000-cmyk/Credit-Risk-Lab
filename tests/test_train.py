import numpy as np
from src.train import choose_threshold, classification_cost, evaluate

def test_missed_default_costs_five():
    y = np.array([1,0])
    p = np.array([0.2,0.8])
    assert classification_cost(y,p,0.5) == 6

def test_good_classification_zero_cost():
    assert classification_cost(np.array([1,0]),np.array([0.9,0.1]),0.5) == 0

def test_threshold_on_grid():
    t = choose_threshold([1,0,1,0], [0.8,0.2,0.7,0.1])
    assert 0.05 <= t <= 0.95

def test_metrics_all_fields():
    m = evaluate(np.array([0,0,1,1]),np.array([0.1,0.2,0.8,0.9]),0.5)
    assert m['roc_auc'] == 1
    assert m['erreur_cout_total'] == 0
