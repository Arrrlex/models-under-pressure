import numpy as np
from sklearn.metrics import roc_curve
from jaxtyping import Float


def tpr_at_fixed_fpr_score(
    y_true: Float[np.ndarray, " batch_size"],
    y_pred: Float[np.ndarray, " batch_size"],
    fpr: float,
) -> float:
    """Calculate TPR at a fixed FPR threshold.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        fpr: Target false positive rate threshold

    Returns:
        TPR value at the specified FPR threshold
    """
    fpr_vals, tpr_vals, thresholds = roc_curve(y_true, y_pred)

    # Find the TPR value at the closest FPR to our target
    idx = np.argmin(np.abs(fpr_vals - fpr))
    return float(tpr_vals[idx])
