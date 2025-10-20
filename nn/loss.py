import numpy as np
from typing import Tuple


def cross_entropy(logits: np.ndarray, targets: np.ndarray
                  ) -> Tuple[float, np.ndarray]:
    """Perform a cross-entropy loss calculation for a batch of logits and
    target labels.

    Args:
        logits (np.ndarray): The predicted logits.
        targets (np.ndarray): The true target labels.

    Returns:
        Tuple[float, np.ndarray]: The computed loss and the gradient with
                                  respect to the logits.
    """
    # Account for numerical stability by subtracting max logits.
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    softmax = exp_z / exp_z.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(softmax[np.arange(n), targets]).mean()
    grad = softmax
    grad[np.arange(n), targets] -= 1
    grad /= n

    return loss, grad


def mse(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Peform mean squared error loss calculation for a batch of logits and
    target labels.

    Args:
        logits (np.ndarray): The predicted logits.
        targets (np.ndarray): The true target labels.

    Returns:
        Tuple[float, np.ndarray]: The computed loss and the gradient with
                                  respect to the logits.
    """
    n = logits.shape[0]
    loss = ((logits - targets) ** 2).mean()
    grad = 2 * (logits - targets) / n

    return loss, grad


def hinge_loss(logits: np.ndarray, targets: np.ndarray
               ) -> Tuple[float, np.ndarray]:
    """Perform hinge loss calculation for a batch of logits and target labels.

    Args:
        logits (np.ndarray): The predicted logits.
        targets (np.ndarray): The true target labels.

    Returns:
        Tuple[float, np.ndarray]: The computed loss and the gradient with
                                  respect to the logits.
    """
    n = logits.shape[0]
    num_classes = logits.shape[1]

    # Convert targets to one-hot encoding.
    targets_one_hot = np.zeros((n, num_classes))
    targets_one_hot[np.arange(n), targets] = 1

    # Compute margins for all classes.
    margins = np.maximum(0, 1 - targets_one_hot * logits)

    # Compute loss.
    loss = margins.mean()

    # Compute the gradient.
    grad = np.zeros_like(logits)
    grad[margins > 0] = -targets_one_hot[margins > 0]
    grad /= n

    return loss, grad
