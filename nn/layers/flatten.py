from .base_layer import Layer
from typing import Optional, Tuple
import numpy as np


class Flatten(Layer):
    """The Flatten Layer, which reshapes the input tensor
    from (B, C, H, W) to (B, C*H*W).
    """

    def __init__(self):
        """Initialize Flatten layer."""
        # Store input shape for backward pass.
        super().__init__()
        self.cache_shape: Optional[Tuple[int, int, int, int]] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform the forward pass of the layer.

        Args:
            x (np.ndarray): The input to the layer.
            training (bool, optional): The flag indicating whether in training
                                       mode. Defaults to True.

        Returns:
            np.ndarray: The output of the layer.
        """
        self.cache_shape = x.shape  # cache input shape for backward pass
        B = x.shape[0]
        y = x.reshape(B, -1)  # flatten all but batch dimension
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        # Ensure that forward has been called.
        assert self.cache_shape is not None, "Must call forward before backward!"

        B, C, H, W = self.cache_shape
        # Unflatten the gradient to the original input shape.
        dx = dout.reshape(B, C, H, W)

        return dx

    def params(self) -> list[np.ndarray]:
        """
        Define the parameters of the layer.

        Returns:
            list[np.ndarray]: The list of parameters.
        """
        return []

    def grads(self) -> list[np.ndarray]:
        """
        Define the gradients of the layer.

        Returns:
            list[np.ndarray]: The list of gradients.
        """
        return []

    def output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute the output shape given the input shape.

        Args:
            input_shape (tuple): The shape of the input.

        Returns:
            tuple: The shape of the output.
        """
        batch_size, channels, height, width = input_shape

        return (batch_size, channels * height * width, 1, 1)
