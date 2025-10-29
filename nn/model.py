from typing import List, Union
import numpy as np
from layers import Layer


class Model:
    """The Model class."""
    def __init__(self, layers: List[Layer]) -> None:
        """
        Initialise the model with a list of layers.

        Args:
            layers (List[Layer]): List of layers in the model.
        """
        self.layers = layers
        self.training = True

    def train(self) -> None:
        """
        Set the model to training mode.
        """
        self.training = True

    def eval(self) -> None:
        """
        Set the model to evaluation mode.
        """
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through all layers.

        Args:
            x (np.ndarray): The input to the model.

        Returns:
            np.ndarray: The output of the model.
        """
        for layer in self.layers:
            x = layer.forward(x, training=self.training)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass through all layers.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self) -> List[np.ndarray]:
        """
        Collect all parameters from all layers.

        Returns:
            List[np.ndarray]: List of all parameters.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        return params

    def grads(self) -> List[np.ndarray]:
        """
        Collect all gradients from all layers.

        Returns:
            List[np.ndarray]: List of all gradients.
        """
        grads = []
        for layer in self.layers:
            grads.extend(layer.grads())
        return grads

    def add_layer(self, layer: Layer) -> None:
        """
        Add a layer to the model.

        Args:
            layer (Layer): The layer to add.
        """
        self.layers.append(layer)

    def remove_layer(self, index_or_name: Union[int, str]) -> None:
        """
        Remove a layer from the model by its index or name.


        Args:
            index_or_name (Union[int, str]): The index or name of the layer to
                                             remove.

        Raises:
            IndexError: If the layer index is out of range.
        """
        if isinstance(index_or_name, int):
            if 0 <= index_or_name < len(self.layers):
                del self.layers[index_or_name]
            else:
                raise IndexError("Layer index out of range")
        else:
            self.layers = [layer for layer in self.layers
                           if layer.name != index_or_name]

    def list_layers(self) -> None:
        """List the layers in the model."""
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: {layer.name if layer.name else 'Unnamed'}")

    def move_layer(self, from_index: int, to_index: int) -> None:
        """
        Move a layer from one position to another.

        Args:
            from_index (int): The current index of the layer.
            to_index (int): The new index for the layer.

        Raises:
            IndexError: If the layer index is out of range.
        """
        if (0 <= from_index < len(self.layers)
                and 0 <= to_index < len(self.layers)):
            layer = self.layers.pop(from_index)

            self.layers.insert(to_index, layer)
        else:
            raise IndexError("Layer index out of range")
