import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from biomedxpro.core.domain import EvaluationMetrics


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> EvaluationMetrics:
    """
    Computes standard classification metrics and fitness.
    """
    # Use float64 for stability and a larger epsilon to avoid log(0) in float32
    epsilon = 1e-7
    y_prob = y_prob.astype(np.float64)
    y_prob_clipped = np.clip(y_prob, epsilon, 1.0 - epsilon)

    # Inverted BCE Fitness
    bce_loss = -np.mean(
        y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped)
    )
    inverted_bce = 1.0 / (1.0 + bce_loss)

    return {
        "inverted_bce": float(inverted_bce),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else 0.5,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
