import numpy as np
from .base_layers import Layer


class Concat(Layer):
    """The Concat Layer, which performs concatenations.

    This class concatenates multiple input arrays along a specified axis.
    """

    def __init__(self, axis: int, name: str = "Concat") -> None:
        """Initialise the Concat Layer.

        Args:
            axis (int): The axis along which to concatenate the inputs.
            name (str, optional): The name of the layer. Defaults to "Concat".
        """
        super().__init__(name)

        self.axis = axis
        self.input_shapes = None

    def forward(self, x: list[np.ndarray], training: bool = True
                ) -> np.ndarray:
        """Perform the forward pass of the layer.

        Args:
            x (list[np.ndarray]): The input to the layer.
            training (bool, optional): The flag indicating whether in training
                                       mode. Defaults to True.

        Returns:
            np.ndarray: The output of the layer.
        """
        self.input_shapes = [input.shape for input in x]

        return np.concatenate(x, axis=self.axis)

    def backward(self, dout: np.ndarray) -> list[np.ndarray]:
        """Perform the backward pass of the layer.

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

    def output_shape(self, input_shapes: list[tuple]) -> tuple:
        """Compute the output shape given the input shape.

        Args:
            input_shapes (list[tuple]): The shapes of the inputs.

        Returns:
            tuple: The shape of the output.
        """
        output_shape = list(input_shapes[0])

        for shape in input_shapes[1:]:
            output_shape[self.axis] += shape[self.axis]

        return tuple(output_shape)
