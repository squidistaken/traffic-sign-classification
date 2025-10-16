# Linear, Conv2D, MaxPool2D, Flatten, BatchNorm2D, Dropout

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Layer(ABC):
    """
    Base class for all layers. Each implementing class should include:
    - forward(x, training=True) -> y: takes input and training bool, returns output
    - backward(dout) -> dx: takes upstream gradient and returns downstream gradient
    - params() -> list of parameters (for optimizers)
    - grads() -> list of gradients (for optimizers)

    Expect np tensors in NCHW format.
    Be careful with caching unnecessary values.
    """

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def params(self) -> list[np.ndarray]:
        pass

    @abstractmethod
    def grads(self) -> list[np.ndarray]:
        pass
