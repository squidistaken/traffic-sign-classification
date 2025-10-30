from .base_layers import Layer2D
from typing import Optional
import numpy as np
from ..utils import image_to_column, column_to_image


class MaxPool2D(Layer2D):
    """The MaxPool2D Layer, which performs 2D max pooling."""
    def __init__(self, pool_size: int, stride: int = 1,
                 padding: int = 0, name: str = "MaxPool2D") -> None:
        """Initialize the MaxPool2D layer.

        Args:
            kernel_size (int): Size of the pooling window. We assume square
                               kernels.
            stride (int, optional): Stride of the pooling operation. Defaults
                                    to 1.
            padding (int, optional): Zero-padding added to both sides of the
                                     input. Defaults to 0.
        """
        if stride is None:
            stride = pool_size
        super().__init__(stride, padding, name)
        self.pool_size = pool_size

        # Cache for the backward pass.
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
        pool_H, pool_W = self.pool_size, self.pool_size

        # Compute the output dimensions.
        out_H = 1 + (H + 2 * self.padding - pool_H) // self.stride
        out_W = 1 + (W + 2 * self.padding - pool_W) // self.stride

        # Convert the input image to columns.
        cols = image_to_column(x, pool_H, self.stride, self.padding)

        # Reshape for the columns for the pooling operation.
        cols_reshaped = cols.reshape(N, C, out_H, out_W, self.pool_size,
                                     self.pool_size)

        # Apply max pooling to each column.
        max_vals = np.max(cols_reshaped, axis=(4, 5))

        # Reshape the output.
        out = max_vals.reshape(N, C, out_H, out_W)

        # Cache the variables needed for the backward pass.
        self.cache = {"x": x, "cols": cols, "cols_reshaped": cols_reshaped}

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the MaxPool2D layer.
        Args:
            dout (np.ndarray): The upstream gradient of shape (N, C, out_H, out_W).
        Returns:
            np.ndarray: The downstream gradient of shape (N, C, H, W).
        """
        x = self.cache["x"]
        cols_reshaped = self.cache["cols_reshaped"]
        N, C, H, W = x.shape
        pool_h = self.pool_size
        pool_w = self.pool_size  # Assuming square pooling window
        stride = self.stride
        padding = self.padding
        N, C, out_H, out_W, _, _ = cols_reshaped.shape

        # Initialize the output gradient
        dx = np.zeros((N, C, H + 2 * padding, W + 2 * padding))

        # For each position in the output
        for n in range(N):
            for c in range(C):
                for i in range(out_H):
                    for j in range(out_W):
                        # Get the pooling window
                        window = cols_reshaped[n, c, i, j]  # shape = (pool_h, pool_w)
                        # Find the position of the max in the window
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        # Add the gradient to the corresponding input position
                        h_start = i * stride
                        w_start = j * stride
                        dx[n, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[n, c, i, j]

        # Remove padding if necessary
        if padding > 0:
            dx = dx[:, :, padding:-padding, padding:-padding]

        return dx

