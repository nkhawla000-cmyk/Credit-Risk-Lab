import numpy as np
from src.train import evaluate, classification_cost


def test_evaluate_confusion_and_cost():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.7, 0.1, 0.9])
    m = evaluate(y, p, 0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["erreur_cout_total"] == 6
    assert classification_cost(y, p, 0.5) == 6
