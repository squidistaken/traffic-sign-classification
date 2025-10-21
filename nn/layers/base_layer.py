from abc import ABC, abstractmethod
import numpy as np


# TODO: Conv2D, MaxPool2D, BatchNorm2D, GlobalAvgPool2D, Sequential
class Layer(ABC):
    """
    Abstract Base Class for Layers.
    NP arrays are expected in NCHW format.
    """

    def __init__(self, name: str = None) -> None:
        """
        Initialize layer parameters and gradients.

        Args:
            name (str, optional): The name of the layer. Defaults to None.
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
    def output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute the output shape given the input shape.

        Args:
            input_shape (tuple): The shape of the input.

        Returns:
            tuple: The shape of the output.
        """
        pass
