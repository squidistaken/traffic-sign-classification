from .base_layers import Layer
from typing import Optional
import numpy as np


class Dropout(Layer):
    """The Dropout Layer, which randomly sets a fraction p of input units to
    zero.

    This class implements dropout, a technique used to prevent overfitting by
    randomly setting a fraction of input units to zero during training.
    """
    def __init__(self, p: float = 0.5, name: str = "Dropout") -> None:
        """Initialise Dropout layer.

        Args:
            p (float, optional): The dropout probability. Defaults to 0.5.
            name (str, optional): The name of the layer. Defaults to "Dropout".
        """
        super().__init__(name)

        assert 0 < p < 1, "Dropout probability must be in (0, 1)"

        self.p = p

        # Store dropout mask for backward pass.
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Perform the forward pass of the layer.

        Args:
            x (np.ndarray): The input to the layer.
            training (bool, optional): The flag indicating whether in training
                                       mode. Defaults to True.

        Returns:
            np.ndarray: The output of the layer.
        """
        if training:
            # Create dropout mask, and scale the activations to keep expected
            # value the same.
            self.mask = (
                (np.random.rand(*x.shape) >= self.p).astype(np.float32)
                / (1.0 - self.p)
            )
            y = x * self.mask

            return y
        else:
            # There is no dropout during evaluation.
            return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform the backward pass of the layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        if self.mask is not None:
            return dout * self.mask
        else:
            # There is no dropout mask stored, so just pass the gradient
            # through.
            return dout

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
        return input_shape
