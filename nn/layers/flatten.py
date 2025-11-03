from .base_layers import Layer
from typing import Optional, Tuple
import numpy as np


class Flatten(Layer):
    """The Flatten Layer, which reshapes the input tensor.

    This class flattens the input tensor by combining the channel, height, and
    width dimensions into a single dimension, while preserving the batch
    dimension.
    """

    def __init__(self, name: str = "Flatten") -> None:
        """Initialize Flatten layer.

        Args:
            name (str, optional): The name of the layer. Defaults to "Flatten".
        """
        super().__init__(name)

        self.cache_shape: Optional[Tuple[int, int, int, int]] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Perform the forward pass of the layer.

        Args:
            x (np.ndarray): The input to the layer.
            training (bool, optional): The flag indicating whether in training
                                       mode. Defaults to True.

        Returns:
            np.ndarray: The output of the layer.
        """
        # Cache the input shape for the backward pass.
        self.cache_shape = x.shape
        B = x.shape[0]

        # Flatten all but the batch dimension.
        y = x.reshape(B, -1)

        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform the backward pass of the layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        # Ensure that forward has been called.
        assert self.cache_shape is not None, (
            "Must call forward before backward!")

        B, C, H, W = self.cache_shape

        # Unflatten the gradient to the original input shape.
        dx = dout.reshape(B, C, H, W)

        return dx

    def params(self) -> dict:
        """Return the learnable parameters of the layer.

        Returns:
            dict: A dictionary mapping parameter names to their values.
        """
        return {}

    def grads(self) -> dict:
        """Return the gradients of the learnable parameters.

        Returns:
            dict: A dictionary mapping parameter names to their gradients.
        """
        return {}

    def output_shape(self, input_shape: tuple) -> tuple:
        """Compute the output shape given the input shape.

        Args:
            input_shape (tuple): The shape of the input.

        Returns:
            tuple: The shape of the output.
        """
        batch_size, channels, height, width = input_shape

        return (batch_size, channels * height * width)
