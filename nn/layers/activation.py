from typing import Dict, Tuple
import numpy as np
from .base_layers import Layer
from ..activations import backward_pass


class Activation(Layer):
    """
    Layer that applies an activation function.
    """
    def __init__(self, activation_fn: callable, name: str = "ActivationLayer", **kwargs):
        """
        Initialize the ActivationLayer.

        Args:
            activation_fn (callable): The activation function to use.
            name (str, optional): The name of the layer. Defaults to "ActivationLayer".
            **kwargs: Additional keyword arguments for the activation function.
        """
        super().__init__(name)
        self.activation = activation_fn
        self.kwargs = kwargs
        self.output = None
        self.grad = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform the forward pass of the layer.
        Args:
            x (np.ndarray): The input to the layer.
            training (bool, optional): The flag indicating whether in training mode. Defaults to True.
        Returns:
            np.ndarray: The output of the layer.
        """
        self.output, self.grad = self.activation(x, **self.kwargs)
        return self.output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the layer.
        Args:
            dout (np.ndarray): The upstream gradient.
        Returns:
            np.ndarray: The downstream gradient.
        """
        return backward_pass(dout, self.grad)

    def params(self) -> Dict[str, np.ndarray]:
        """
        Return the learnable parameters of the layer.
        Returns:
            Dict[str, np.ndarray]: A dictionary mapping parameter names to their values.
        """
        return {}  # Activation layers typically have no learnable parameters

    def grads(self) -> Dict[str, np.ndarray]:
        """
        Return the gradients of the learnable parameters.
        Returns:
            Dict[str, np.ndarray]: A dictionary mapping parameter names to their gradients.
        """
        return {}  # Activation layers typically have no learnable parameters

    def output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Compute the output shape given the input shape.
        Args:
            input_shape (Tuple): The shape of the input.
        Returns:
            Tuple: The shape of the output.
        """
        return input_shape  # Activation layers do not change the shape of the input
