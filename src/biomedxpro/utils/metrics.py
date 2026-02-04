# src/biomedxpro/utils/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from biomedxpro.core.domain import EvaluationMetrics


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> EvaluationMetrics:
    """
    Computes standard classification metrics.
    Supports both Binary (2 classes) and Multi-Class (N > 2) classification.

    Args:
        y_true: Ground truth labels (N_samples,).
        y_pred: Predicted labels (N_samples,).
        y_prob: Predicted probability distribution.
                Binary: (N_samples,) or (N_samples, 1) or (N_samples, 2)
                Multi-Class: (N_samples, N_classes)

    Returns:
        EvaluationMetrics dict.
    """
    # 1. Determine Classification Mode
    # If y_prob has shape (N, N_classes) where N_classes > 2, it is multi-class.
    is_multiclass = y_prob.ndim == 2 and y_prob.shape[1] > 2

    # 2. Compute F1 & Accuracy (Label-based metrics)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # 3. Compute ROC-AUC (Probability-based metric)
    try:
        if is_multiclass:
            # Multi-class AUC requires 'ovr' (One-vs-Rest) and full probability matrix
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        else:
            # Binary Case: roc_auc_score expects probabilities of the POSITIVE class.
            # If input is (N, 2), take the second column.
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob_auc = y_prob[:, 1]
            else:
                y_prob_auc = y_prob.ravel()

            auc = roc_auc_score(y_true, y_prob_auc)
    except ValueError:
        # Handle edge cases (e.g., only one class present in y_true batch)
        auc = 0.5

    # 4. Compute Inverted BCE (Log Loss)
    # We invert it because the evolutionary algorithm MAXIMIZES fitness.
    # Use 1/(1+loss) to keep result bounded in [0, 1] range.
    try:
        # log_loss handles both binary and multi-class automatically
        loss = log_loss(y_true, y_prob)
        inverted_bce = 1.0 / (1.0 + loss)
    except ValueError:
        inverted_bce = 0.0

    # 5. --- MARGIN SCORE IMPLEMENTATION ---
    # Goal: Calculate mean( Prob(Correct Class) - Max_Prob(Incorrect Class) )

    # A. Get indices for the True Class
    n_samples = len(y_true)
    row_indices = np.arange(n_samples)

    # B. Extract probability of the correct class
    # y_prob[0, label_0], y_prob[1, label_1], ...
    p_correct = y_prob[row_indices, y_true]

    # C. Find max probability of incorrect classes
    # We create a copy so we don't mutate the original array used for AUC
    y_prob_masked = y_prob.copy()

    # Mask the true class by setting it to -1.0 (since probs are [0,1], -1 is always lower)
    y_prob_masked[row_indices, y_true] = -1.0

    # Now the max() along the axis will naturally find the runner-up
    p_max_incorrect = np.max(y_prob_masked, axis=1)

    # D. Compute Margin (-1.0 to 1.0)
    # High positive = Confident Correct
    # Negative = Confident Incorrect
    # Near Zero = Confused / Decision Boundary
    sample_margins = p_correct - p_max_incorrect
    margin_score = float(np.mean(sample_margins))

    # E. --- PER-CLASS MARGINS ---
    # Group margins by class to provide per-class report card
    num_classes = y_prob.shape[1]
    per_class_margins = []

    for class_idx in range(num_classes):
        # Find all samples belonging to this class
        indices = np.where(y_true == class_idx)[0]

        if len(indices) > 0:
            # Average margin for this class
            score = float(np.mean(sample_margins[indices]))
        else:
            # Handle edge case: This class wasn't in the batch
            score = 0.0

        per_class_margins.append(score)

    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "inverted_bce": float(inverted_bce),
        "f1_macro": float(f1_macro),
        "accuracy": float(accuracy),
        "auc": float(auc),
        "f1_weighted": float(f1_weighted),
        "margin_score": margin_score,
        "per_class_margins": per_class_margins,
        "confusion_matrix": cm,
    }
