import numpy as np
from .base_layers import Layer


class Concat(Layer):
    """The Concat Layer, which performs concatenations."""
    def __init__(self, axis: int) -> None:
        """Initialise the Concat Layer.

        Args:
            axis (int):  The axis along which to concatenate the inputs.
        """
        super().__init__()
        self.axis = axis
        self.input_shapes = None

    def forward(self, x: list[np.ndarray],
                training: bool = True) -> np.ndarray:
        """
        Perform the forward pass of the layer.

        Args:
            x (np.ndarray): The input to the layer.
            training (bool, optional): The flag indicating whether in training
                                       mode. Defaults to True.

        Returns:
            np.ndarray: The output of the layer.
        """
        self.input_shapes = [input.shape for input in x]
        return np.concatenate(x, axis=self.axis)

    def backward(self, dout: np.ndarray) -> list[np.ndarray]:
        """
        Perform the backward pass of the layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            list[np.ndarray]: List of gradients corresponding to each input.
        """
        gradients = []
        start = 0
        for shape in self.input_shapes:
            slice_obj = [slice(None)] * dout.ndim
            slice_obj[self.axis] = slice(start, start + shape[self.axis])
            gradients.append(dout[tuple(slice_obj)])
            start += shape[self.axis]
        return gradients

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

    def output_shape(self, input_shapes: list[tuple]) -> tuple:
        """
        Compute the output shape given the input shape.

        Args:
            input_shape (tuple): The shape of the input.

        Returns:
            tuple: The shape of the output.
        """
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)
