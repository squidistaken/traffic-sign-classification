from .base_layers import Layer
import numpy as np


class Sequential(Layer):
    """The Sequential Layer, which chains multiple layers together."""
    def __init__(self, layers: list[Layer]):
        """Initialize Sequential layer.

        Args:
            layers (list[Layer]): The list of layers to chain.
        """
        super().__init__()

        self.layers = layers

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
        for layer in self.layers:
            x = layer.forward(x, training)

        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def params(self) -> list[np.ndarray]:
        """
        Define the parameters of the layer.

        Returns:
            list[np.ndarray]: The list of parameters.
        """
        params = []

        for layer in self.layers:
            params.extend(layer.params())

        return params

    def grads(self) -> list[np.ndarray]:
        """
        Define the gradients of the layer.

        Returns:
            list[np.ndarray]: The list of gradients.
        """
        grads = []

        for layer in self.layers:
            grads.extend(layer.grads())

        return grads

    def output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute the output shape given the input shape.

        Args:
            input_shape (tuple): The shape of the input.

        Returns:
            tuple: The shape of the output.
        """
        shape = input_shape

        for layer in self.layers:
            shape = layer.output_shape(shape)

        return shape
