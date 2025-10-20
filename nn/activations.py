import numpy as np
from typing import Tuple, Union


# region Forward Pass Activations
def relu(x: Union[float, np.ndarray], alpha: float = 0.0
         ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute the forward pass and gradient of the Rectified Linear Unit (ReLU)
    activation function. If alpha is non-zero, it behaves like a Leaky ReLU.

    Args:
        x (Union[float, np.ndarray]): The input value or array.
        alpha (float): The slope for the negative part of the function.
                       Defaults to 0.0

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]: The tuple
                                                                   containing
                                                                   the output
                                                                   of the
                                                                   (leaky) ReLU
                                                                   function and
                                                                   its gradient
                                                                   at x.
    """
    if alpha != 0.0:
        # Function as leaky ReLU.
        output = np.where(x > 0, x, alpha * x)
        grad = np.where(x > 0, 1.0, alpha)
    else:
        # Function as standard ReLU.
        output = np.maximum(0.0, x)
        grad = np.where(x > 0, 1.0, 0.0)

    return output, grad


def tanh(x: Union[float, np.ndarray]
         ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute the forward pass and gradient of the Tanh activation function.

    Args:
        x (Union[float, np.ndarray]): The input value or array.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]: The tuple
                                                                   containing
                                                                   the output
                                                                   of the Tanh
                                                                   function and
                                                                   its gradient
                                                                   at x.
    """
    output = np.tanh(x)
    grad = 1 - output ** 2

    return output, grad


def sigmoid(x: Union[float, np.ndarray]
            ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Compute the forward pass and gradient of the Sigmoid activation function.

    Args:
        x (Union[float, np.ndarray]): The input value or array.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]: The tuple
                                                                   containing
                                                                   the output
                                                                   of the
                                                                   Sigmoid
                                                                   function and
                                                                   its gradient
                                                                   at x.
    """
    output = 1 / (1 + np.exp(-x))
    grad = output * (1 - output)

    return output, grad


def softmax(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the forward pass and gradient of the Softmax activation function.

    Args:
        x (np.ndarray): The input array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The tuple containing the output of the
                                       Softmax function and its gradient at x.
    """
    # Ensure numerical stability by subtracting max from input.
    exp_x = np.exp(x - np.max(x))
    sum_exp = np.sum(exp_x)
    output = exp_x / sum_exp
    # This is a simplification; the actual gradient is a Jacobian.
    grad = output * (1 - output)

    return output, grad
# endregion


# region Backward Pass Activations
def backward_pass(dout: Union[float, np.ndarray],
                  grad: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the backward pass for any given activation function.

    Args:
        dout (Union[float, np.ndarray]): The gradient of the loss with respect
                                         to the output.
        grad (Union[float, np.ndarray]): The gradient of the activation
                                         function at x.

    Returns:
        Union[float, np.ndarray]: The gradient of the loss with respect to the
                                  input.
    """
    return dout * grad
# endregion
