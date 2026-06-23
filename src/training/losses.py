"""
Custom loss functions for the training pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces loss contribution from easy examples, focusing on hard negatives.
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class WeightedBCELoss(nn.Module):
    """Binary cross-entropy with configurable positive class weight."""

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight.to(logits.device)
        )


def get_loss_fn(name: str, **kwargs) -> nn.Module:
    """Factory for loss functions.

    Parameters
    ----------
    name : str
        One of: 'bce', 'mse', 'focal'.
    """
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "mse":
        return nn.MSELoss()
    elif name == "focal":
        return FocalLoss(gamma=kwargs.get("gamma", 2.0), alpha=kwargs.get("alpha", 0.25))
    elif name == "weighted_bce":
        return WeightedBCELoss(pos_weight=kwargs.get("pos_weight", 1.0))
    else:
        raise ValueError(f"Unknown loss: {name}")
