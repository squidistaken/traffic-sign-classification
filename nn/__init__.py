from .activations import relu, softmax, sigmoid, tanh
from .checks import (
    check_data_shape,
    check_data_range,
    check_model_convergence,
    check_model_performance,
)
from .loss import cross_entropy
from .optim import SGD, Adam, Momentum
from .sched import StepLRScheduler, WarmupLRScheduler, CosineLRScheduler
from .utils import image_to_column, column_to_image, one_hot_encode, compute_accuracy
from . import layers
import numpy as np


def xavier_init(shape: tuple) -> np.ndarray:
    """
    Xavier (Glorot) initialisation for weights.
    Args:
        shape (tuple): The shape of the weight matrix.
    Returns:
        np.ndarray: The initialised weights.
    """
    if len(shape) == 2:
        n_in, n_out = shape
    else:
        # For convolutional layers, consider the number of input channels and kernel size
        n_in = shape[1] * np.prod(shape[2:])
        n_out = shape[0]
    var = 2.0 / (n_in + n_out)
    return np.random.normal(0.0, np.sqrt(var), shape)


def he_init(shape: tuple) -> np.ndarray:
    """
    He initialization for weights.
    Args:
        shape (tuple): The shape of the weight matrix.
    Returns:
        np.ndarray: The initialised weights.
    """
    if len(shape) == 2:
        n_in, _ = shape
    else:
        # For convolutional layers, consider the number of input channels and kernel size
        n_in = shape[1] * np.prod(shape[2:])
    var = 2.0 / n_in
    return np.random.normal(0.0, np.sqrt(var), shape)



__all__ = [
    "relu",
    "softmax",
    "sigmoid",
    "tanh",
    "check_data_shape",
    "check_data_range",
    "check_model_convergence",
    "check_model_performance",
    "cross_entropy",
    "SGD",
    "Adam",
    "Momentum",
    "StepLRScheduler",
    "WarmupLRScheduler",
    "CosineLRScheduler",
    "image_to_column",
    "column_to_image",
    "one_hot_encode",
    "compute_accuracy",
    "layers",
]
