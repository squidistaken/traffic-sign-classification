import numpy as np
from typing import Optional
from .base_layers import Layer


class Linear(Layer):
    """
    The Linear (Fully Connected) Layer, which performs an affine
    transformation on the input data.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        W: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        name: str = "Linear",
    ) -> None:
        """
        Initialize the Linear layer. If the weight matrix W or bias b
        are not provided, they are initialized randomly.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            W (Optional[np.ndarray], optional): The weight matrix.
                                                Defaults to None.
            b (Optional[np.ndarray], optional): The bias vector.
                                                Defaults to None.
        """
        super().__init__(name)

        if W is None:
            # The initialization uses He initialization for weights. This is
            # particularly effective when using ReLU activations.
            self.W = np.random.randn(in_features, out_features) * np.sqrt(
                2 / in_features
            )
        else:
            self.W = W
        if b is None:
            self.b = np.zeros(out_features)
        else:
            self.b = b

        # Store these just to be sure
        self.in_features = in_features
        self.out_features = out_features

        self.grad_weights = np.zeros_like(self.W)
        self.grad_biases = np.zeros_like(self.b)

        # Store input for backward pass.
        self.cache_x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform the forward pass of the Linear layer.

        Args:
            x (np.ndarray): The input to the layer.
            training (bool, optional): The flag indicating whether in training
                                       mode. Defaults to True.

        Returns:
            np.ndarray: The output of the layer.
        """
        # Cache the input for use in the backward pass.
        self.cache_x = x
        y = x.dot(self.W) + self.b
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the Linear layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        # Ensure that forward has been called.
        assert self.cache_x is not None, "Must call forward before backward!"

        x = self.cache_x

        # The batch is averaged over in gradient calculations.
        self.grad_weights = (x.T @ dout) / x.shape[0]
        self.grad_biases = dout.mean(axis=0)
        dx = dout @ self.W.T

        return dx

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"dW": self.grad_weights, "db": self.grad_biases}


    def output_shape(self, input_shape: tuple) -> tuple:
        """
        Compute the output shape given the input shape.

        Args:
            input_shape (tuple): The shape of the input.

        Returns:
            tuple: The shape of the output.
        """
        return (input_shape[0], self.out_features, 1, 1)
