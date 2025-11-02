from .base_layers import Layer2D
from typing import Optional, Callable
import numpy as np


class BatchNorm2D(Layer2D):
    """The BatchNorm2D Layer, which performs 2D batch normalization."""
    def __init__(
        self,
        num_channels: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
        name: str = "BatchNorm2D",
        weight_init: Optional[Callable] = None
    ) -> None:
        """Initialize the BatchNorm2D layer.
        Args:
            num_features (int): Number of features (channels) in the input.
            momentum (float, optional): The momentum for running mean and
                                        variance. Defaults to 0.9.
            epsilon (float, optional): The small constant to avoid division by
                                       zero. Defaults to 1e-5.
            weight_init (Optional[Callable]): A function to initialize the weights.
        """
        super().__init__(name, weight_init)
        self.num_channels = num_channels
        self.momentum = momentum
        self.epsilon = epsilon
        # Initialize the scale and shift parameters.
        if weight_init is not None:
            self.gamma = weight_init((num_channels,))
        else:
            self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)
        # Initialize gradients.
        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)
        # Initialize the running mean and variance.
        self.running_mean = np.zeros(num_channels)
        self.running_var = np.ones(num_channels)
        # Cache the backward pass variables.
        self.cache: Optional[dict] = None

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
        N, C, H, W = x.shape

        # Compute the mean and variance over the batch and spatial dimensions.
        if training:
            mean = np.mean(x, axis=(0, 2, 3))
            var = np.var(x, axis=(0, 2, 3))

            # Update the running mean and variance.
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize the input.
        x_norm = (x - mean.reshape(1, C, 1, 1)) / np.sqrt(
            var.reshape(1, C, 1, 1) + self.epsilon
        )

        # Scale and shift the normalized input.
        out = self.gamma.reshape(1, C, 1, 1) * x_norm + self.beta.reshape(1, C, 1, 1)

        # Cache variables for backward pass.
        self.cache = {"x": x, "x_norm": x_norm, "mean": mean, "var": var}

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the layer.
        Args:
            dout (np.ndarray): The upstream gradient.
        Returns:
            np.ndarray: The downstream gradient.
        """
        x = self.cache["x"]
        x_norm = self.cache["x_norm"]
        mean = self.cache["mean"]
        var = self.cache["var"]
        N, C, _, _ = x.shape
        # Compute gradients.
        self.grad_gamma = np.sum(dout * x_norm, axis=(0, 2, 3))
        self.grad_beta = np.sum(dout, axis=(0, 2, 3))
        dx_norm = dout * self.gamma.reshape(1, C, 1, 1)
        dvar = np.sum(
            dx_norm
            * (x - mean.reshape(1, C, 1, 1))
            * -0.5
            * (var.reshape(1, C, 1, 1) + self.epsilon) ** -1.5,
            axis=(0, 2, 3),
        )
        dmean = np.sum(
            dx_norm * -1 / np.sqrt(var.reshape(1, C, 1, 1) + self.epsilon),
            axis=(0, 2, 3),
        ) + dvar * -2 * np.mean(x - mean.reshape(1, C, 1, 1), axis=(0, 2, 3))
        dx = (
            dx_norm / np.sqrt(var.reshape(1, C, 1, 1) + self.epsilon)
            + dvar.reshape(1, C, 1, 1) * 2 * (x - mean.reshape(1, C, 1, 1)) / N
            + dmean.reshape(1, C, 1, 1) / N
        )
        return dx

    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self):
        return {"dgamma": self.grad_gamma, "dbeta": self.grad_beta}


    def output_shape(
        self, input_shape: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        """Compute the output shape given the input shape.

        Args:
            input_shape (Tuple[int, int, int, int]): The shape of the input.

        Returns:
            Tuple[int, int, int, int]: The shape of the output.
        """
        return input_shape
