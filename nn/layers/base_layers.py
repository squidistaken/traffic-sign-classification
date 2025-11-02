from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


# region Layer
class Layer(ABC):
    """
    Abstract Base Class for Layers. NP arrays are expected in NCHW format.
    """

    def __init__(self, name: str = "Layer") -> None:
        """
        Initialize layer parameters and gradients.

        Args:
            name (str, optional): The name of the layer. Defaults to "Layer".
        """
        self.name = name

    @abstractmethod
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
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        pass

    @abstractmethod
    def params(self) -> list[np.ndarray]:
        """
        Define the parameters of the layer.

        Returns:
            list[np.ndarray]: The list of parameters.
        """
        pass

    @abstractmethod
    def grads(self) -> list[np.ndarray]:
        """
        Define the gradients of the layer.

        Returns:
            list[np.ndarray]: The list of gradients.
        """
        pass

    @abstractmethod
    def output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Compute the output shape given the input shape.

        Args:
            input_shape (Tuple): The shape of the input.

        Returns:
            Tuple: The shape of the output.
        """
        pass


# endregion


# region Layer2D
class Layer2D(Layer):
    """
    Abstract Base Class for 2D Layers. NP arrays are expected in NCHW format.
    """

    def __init__(
        self, stride: int = 1, padding: int = 0, name: str = "Layer2D"
    ) -> None:
        """
        Initialize the 2D layer.

        Args:
            stride (int, optional): The stride of the layer. Defaults to 1.
            padding (int, optional): The padding of the layer. Defaults to 0.
        """
        super().__init__(name)
        self.stride = stride
        self.padding = padding

    def output_shape(
        self, input_shape: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Compute the output shape given the input shape.

        Args:
            input_shape (Tuple[int, int, int, int]): The shape of the input.

        Returns:
            Tuple[int, int, int, int]: The shape of the output.
        """
        N, C, H, W = input_shape
        out_H = 1 + (H + 2 * self.padding - 1) // self.stride
        out_W = 1 + (W + 2 * self.padding - 1) // self.stride

        return (N, C, out_H, out_W)

    def pad_input(self, x: np.ndarray) -> np.ndarray:
        """
        Pad the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The padded input array.
        """
        if self.padding > 0:
            return np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            return x

    def unpad_input(self, x: np.ndarray) -> np.ndarray:
        """
        Unpad the input array.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The unpadded input array.
        """
        if self.padding > 0:
            return x[:, :, self.padding : -self.padding, self.padding : -self.padding]
        else:
            return x


# endregion
