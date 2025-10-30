from .base_layers import Layer2D
import numpy as np
from typing import Optional


class GlobalAvgPool2D(Layer2D):
    """The GlobalAvgPool2D Layer, which performs global average pooling."""
    def __init__(self, name: str = "GlobalAvgPool2D") -> None:
        """Initialize the GlobalAvgPool2D layer."""
        super().__init__(name, stride=1, padding=0)

        # Cache for backward pass.
        self.cache: Optional[dict] = None

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
        N, C, H, W = x.shape

        # Compute the global average pooling, by averaging over height and
        # width.
        out = np.mean(x, axis=(2, 3))

        # Cache the input for backward pass.
        self.cache = {'input_shape': x.shape}

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        input_shape = self.cache['input_shape']
        N, C, H, W = input_shape

        # Compute the gradient with respect to the input.
        dx = np.repeat(dout.repeat(N, C, 1, 1), H*W, axis=2)
        dx = dx.reshape(N, C, H, W) / (H * W)

        return dx

    def output_shape(self, input_shape):
        """
        Compute the output shape given the input shape.

        Args:
            input_shape (Tuple[int, int, int, int]): The shape of the input.

        Returns:
            Tuple[int, int, int, int]: The shape of the output.
        """
        N, C, H, W = input_shape

        return (N, C, 1, 1)
