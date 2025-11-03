from .base_layers import Layer2D
from typing import Optional, Callable
import numpy as np
from ..utils import image_to_column, column_to_image


class Conv2D(Layer2D):
    """The Conv2D Layer, which performs a 2D convolution operation.

    This class implements a 2D convolution operation, which applies a set of
    filters to the input to extract features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        name: str = "Conv2D",
        weight_init: Optional[Callable] = None,
    ) -> None:
        """Initialise Conv2D layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel. We assume square
                               kernels.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Zero-padding added to both sides of the
                                     input. Defaults to 0.
            name (str, optional): The name of the layer. Defaults to "Conv2D".
            weight_init (Optional[Callable]): A function to initialise the
                                              weights.
        """
        super().__init__(stride, padding, name, weight_init)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialise the weights and biases.
        if weight_init is not None:
            self.weights = weight_init(
                (out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            self.weights = (
                np.random.randn(out_channels, in_channels, kernel_size,
                                kernel_size)
                * 0.01
            )

        self.biases = np.zeros(out_channels)

        # Gradients.
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

        # Cache for backward pass.
        self.cache: Optional[dict] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Perform the forward pass of the layer.

        Args:
            x (np.ndarray): The input to the layer.
            training (bool, optional): The flag indicating whether in training
                                       mode. Defaults to True.

        Returns:
            np.ndarray: The output of the layer.
        """
        N, C, H, W = x.shape
        F, _, HH, WW = self.weights.shape

        # Compute the output dimensions.
        out_H = 1 + (H + 2 * self.padding - HH) // self.stride
        out_W = 1 + (W + 2 * self.padding - WW) // self.stride

        # Convert the input image to columns.
        cols = image_to_column(x, HH, self.stride, self.padding)

        # Reshape the weights to rows.
        weights_matrix = self.weights.reshape(F, -1)
        # Perform the matrix multiplication.

        out = weights_matrix @ cols + self.biases.reshape(-1, 1)

        # Reshape the output.
        out = out.reshape(F, N, out_H, out_W).transpose(1, 0, 2, 3)

        # Cache the input and columns for backward pass.
        self.cache = {"x": x, "cols": cols}

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform the backward pass of the layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        x = self.cache["x"]
        cols = self.cache["cols"]
        N, C, H, W = x.shape
        F, _, HH, WW = self.weights.shape
        _, _, out_H, out_W = dout.shape

        # Compute the gradient with respect to the biases.
        self.grad_biases = np.sum(dout, axis=(0, 2, 3))

        # Compute the gradient with respect to the weights.
        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
        self.grad_weights = dout_reshaped @ cols.T
        self.grad_weights = self.grad_weights.reshape(F, C, HH, WW)

        # Compute the gradient with respect to the input.
        weights_matrix = self.weights.reshape(F, -1)
        dx_cols = weights_matrix.T @ dout_reshaped
        dx = column_to_image(dx_cols, x.shape, HH, self.stride, self.padding)

        return dx

    def params(self) -> dict:
        """Return the learnable parameters of the layer.

        Returns:
            dict: A dictionary mapping parameter names to their values.
        """
        return {"W": self.weights, "b": self.biases}

    def grads(self) -> dict:
        """Return the gradients of the learnable parameters.

        Returns:
            dict: A dictionary mapping parameter names to their gradients.
        """
        return {"dW": self.grad_weights, "db": self.grad_biases}

    def output_shape(self, input_shape: tuple) -> tuple:
        """Compute the output shape given the input shape.

        Args:
            input_shape (tuple): The shape of the input.

        Returns:
            tuple: The shape of the output.
        """
        batch_size, channels, height, width = input_shape
        out_height = (1 + (height + 2 * self.padding - self.kernel_size)
                      // self.stride)
        out_width = (1 + (width + 2 * self.padding - self.kernel_size)
                     // self.stride)
        return (batch_size, self.out_channels, out_height, out_width)
