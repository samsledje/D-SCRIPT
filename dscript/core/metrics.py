"""Classification metrics for interaction predictions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


def interaction_metrics(p_hat: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """
    Same definitions as the original training code. Precision and recall are
    "soft": computed from probabilities, not thresholded predictions.
    """
    p_hat, y = torch.as_tensor(p_hat).double(), torch.as_tensor(y).double()
    tp = torch.sum(y * p_hat).item()
    pr = tp / max(torch.sum(p_hat).item(), 1e-12)
    re = tp / max(torch.sum(y).item(), 1e-12)
    return {
        "loss": F.binary_cross_entropy(p_hat, y).item(),
        "accuracy": ((p_hat > 0.5).double() == y).double().mean().item(),
        "mse": torch.mean((y - p_hat) ** 2).item(),
        "precision": pr,
        "recall": re,
        "f1": 2 * pr * re / (pr + re) if pr + re > 0 else 0.0,
        "aupr": float(average_precision_score(y.numpy(), p_hat.numpy())) if y.sum() > 0 else 0.0,
        "n": len(y),
    }
