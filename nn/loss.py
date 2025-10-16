# CrossEntropy, MSE (for tests)
# purpose: take logits and targets, return loss and gradients

import numpy as np
from typing import Tuple


def cross_entropy_with_logits(
    logits: np.ndarray,  # (B, num_classes)
    targets: np.ndarray,  # (B,)
) -> Tuple[float, np.ndarray]:
    """
    For one batch, takes in all the logits and the target class indices. Returns a tuple (loss, grad) where
    loss is the cross-entropy loss (B,) and grad (B, num_classes) is the gradient of the loss with respect to the logits.
    """
    z = logits - logits.max(axis=1, keepdims=True)  # for numerical stability
    exp_z = np.exp(z)
    softmax = exp_z / exp_z.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(softmax[np.arange(n), targets]).mean()

    grad = softmax
    grad[np.arange(n), targets] -= 1
    grad /= n
    return loss, grad
