# Linear, Conv2D, MaxPool2D, Flatten, BatchNorm2D, Dropout

from abc import ABC, abstractmethod
from typing import Optional, Tuple
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
    def __init__(self):
        pass

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


class Linear(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        W: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
    ):
        """
        Fully connected layer. Input shape (B, in_features), output shape (B, out_features).
        Weights W shape (in_features, out_features), bias b shape (out_features,).
        If W or b are not provided, they are initialized randomly (W with small values, b with zeros).
        """
        if W is None:
            self.W = np.random.randn(in_features, out_features) * np.sqrt(
                2 / in_features
            )  # He initialization, works well with ReLU
        else:
            self.W = W
        if b is None:
            self.b = np.zeros(out_features)
        else:
            self.b = b

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.cache_x: Optional[np.ndarray] = None  # to store input for backward pass

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass for linear layer.
        x shape: (B, in_features)
        returns y shape: (B, out_features)
        """
        self.cache_x = x  # cache input for backward pass
        y = x.dot(self.W) + self.b  # (B, out_features)
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for linear layer.
        dout shape: (B, out_features)
        returns dx shape: (B, in_features)
        """
        assert (
            self.cache_x is not None
        ), "Must call forward before backward"  # for static type checker
        x = self.cache_x  # (B, in_features)
        # (in_features, out_features), averaged over batch
        self.dW = (x.T @ dout) / x.shape[0]
        self.db = dout.mean(axis=0)  # (out_features,), averaged over batch
        dx = dout @ self.W.T  # (B, in_features)
        return dx

    def params(self) -> list[np.ndarray]:
        """
        Returns a list of parameters for the layer.
        """
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        """
        Returns a list of gradients for the layer.
        """
        return [self.dW, self.db]


class Flatten(Layer):
    def __init__(self):
        """
        Flattens input from (B, C, H, W) to (B, C*H*W).
        """
        self.cache_shape: Optional[Tuple[int, int, int, int]] = (
            None  # to store input shape
        )

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass for flatten layer.
        x shape: (B, C, H, W)
        returns y shape: (B, C*H*W)
        """
        self.cache_shape = x.shape  # cache input shape for backward pass
        B = x.shape[0]
        y = x.reshape(B, -1)  # flatten all but batch dimension
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for flatten layer.
        dout shape: (B, C*H*W)
        returns dx shape: (B, C, H, W)
        """
        assert (
            self.cache_shape is not None
        ), "Must call forward before backward"  # for static type checker
        B, C, H, W = self.cache_shape
        dx = dout.reshape(B, C, H, W)  # unflatten
        return dx

    def params(self) -> list[np.ndarray]:
        """
        Flatten layer has no parameters.
        """
        return []

    def grads(self) -> list[np.ndarray]:
        """
        Flatten layer has no gradients.
        """
        return []


class Dropout(Layer):
    def __init__(self, p: float = 0.5):
        """
        Dropout layer. During training, randomly sets a fraction p of input units to zero.
        During evaluation, does nothing.
        """
        assert 0 < p < 1, "Dropout probability must be in (0, 1)"
        self.p = p
        self.mask: Optional[np.ndarray] = None  # to store dropout mask

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass for dropout layer.
        x shape: (B, C, H, W) or (B, features)
        returns y shape: same as x
        """
        if training:
            # dropout mask (0 if dropped, 1 if kept)
            self.mask = (np.random.rand(*x.shape) >= self.p).astype(np.float32) / (
                1.0 - self.p
            )  # scale to keep expected value the same
            y = x * self.mask
            return y
        else:
            return x  # no dropout during evaluation

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for dropout layer.
        dout shape: same as input x
        """
        if self.mask is not None:
            return dout * self.mask / (1 - self.p)
        else:
            return dout  # no dropout during evaluation

    def params(self) -> list[np.ndarray]:
        """
        Dropout layer has no parameters.
        """
        return []

    def grads(self) -> list[np.ndarray]:
        """
        Dropout layer has no gradients.
        """
        return []


# todo: Conv2D, MaxPool2D, BatchNorm2D, GlobalAvgPool2D, Sequential
