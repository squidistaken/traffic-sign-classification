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


def class_balanced_cross_entropy(
    logits: np.ndarray, targets: np.ndarray, class_counts: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Compute class-balanced cross-entropy loss and its gradient.

    Args:
        logits (np.ndarray): The predicted logits.
        targets (np.ndarray): The true target labels.
        class_counts (np.ndarray): The counts of each class in the dataset.

    Returns:
        Tuple[float, np.ndarray]: The computed loss and the gradient with
                                  respect to the logits.
    """
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    softmax = exp_z / exp_z.sum(axis=1, keepdims=True)
    n = logits.shape[0]

    # Compute class frequencies.
    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()

    # Compute loss.
    sample_weights = class_weights[targets]
    log_probs = -np.log(softmax[np.arange(n), targets])
    loss = (sample_weights * log_probs).sum() / sample_weights.sum()

    # Compute gradient.
    grad = softmax
    grad[np.arange(n), targets] -= 1
    grad *= sample_weights[:, np.newaxis]
    grad /= sample_weights.sum()

    return loss, grad


def mse(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute the mean squared error loss and its gradient.

    Args:
        logits (np.ndarray): The predicted logits.
        targets (np.ndarray): The true target labels.

    Returns:
        Tuple[float, np.ndarray]: The computed loss and the gradient with
                                  respect to the logits.
    """
    batch_size, _ = logits.shape

    # Convert integer labels to one-hot.
    one_hot = np.zeros_like(logits)
    one_hot[np.arange(batch_size), targets] = 1.0

    # Compute MSE loss.
    loss = np.mean((logits - one_hot) ** 2)
    # Compute gradient.

    grad_output = 2 * (logits - one_hot) / batch_size
    return loss, grad_output
