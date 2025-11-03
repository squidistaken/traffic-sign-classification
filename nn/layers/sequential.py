from .base_layers import Layer
import numpy as np


class Sequential(Layer):
    """The Sequential Layer, which chains multiple layers together.

    This class allows the sequential composition of multiple layers, where the
    output of one layer is the input to the next.
    """
    def __init__(self, layers: list[Layer], name: str = "Sequential"):
        """Initialise Sequential layer.

        Args:
            layers (list[Layer]): The list of layers to chain.
            name (str, optional): The name of the layer. Defaults to
                                  "Sequential".
        """
        super().__init__(name)

        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Perform the forward pass of the layer.

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
        """Perform the backward pass of the layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def params(self) -> dict:
        """Return the learnable parameters of the layer.

        Returns:
            dict: A dictionary mapping parameter names to their values.
        """
        params = {}

        for i, layer in enumerate(self.layers):
            for name, value in layer.params().items():
                params[f"layer{i}_{name}"] = value

        return params

    def grads(self) -> dict:
        """Return the gradients of the learnable parameters.

        Returns:
            dict: A dictionary mapping parameter names to their gradients.
        """
        grads = {}

        for i, layer in enumerate(self.layers):
            for name, value in layer.grads().items():
                grads[f"layer{i}_{name}"] = value

        return grads

    def output_shape(self, input_shape: tuple) -> tuple:
        """Compute the output shape given the input shape.

        Args:
            input_shape (tuple): The shape of the input.

        Returns:
            tuple: The shape of the output.
        """
        shape = input_shape

        for layer in self.layers:
            shape = layer.output_shape(shape)

        return shape
