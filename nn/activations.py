# ReLU, LeakyReLU, Tanh, Sigmoid, Softmax

from typing import Tuple


def relu_forward(x: float) -> Tuple[float, float]:
    """
    Rectified Linear Unit activation function.
    Returns a tuple (output, grad) where output is the ReLU of x and grad is the gradient of ReLU at x.
    """
    return max(0.0, x), 1.0 if x > 0 else 0.0


def relu_backward(dout: float, grad: float) -> float:
    """
    Backward pass for ReLU activation function.
    """
    return dout * grad
