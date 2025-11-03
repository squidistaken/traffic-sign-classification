from .base_layers import Layer2D
from typing import Optional
import numpy as np
from ..utils import image_to_column, column_to_image


class MaxPool2D(Layer2D):
    """The MaxPool2D Layer, which performs 2D max pooling.

    This class implements 2D max pooling, which down-samples the input by
    taking the maximum value over a specified window size.
    """
    def __init__(
        self,
        pool_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
        name: str = "MaxPool2D",
    ) -> None:
        """Initialise the MaxPool2D layer.

        Args:
            pool_size (int): The size of the pooling window.
            stride (Optional[int], optional): The stride of the pooling.
                                              Defaults to pool_size.
            padding (int, optional): The padding added to both sides of the
                                     input. Defaults to 0.
            name (str, optional): The name of the layer. Defaults to
                                  "MaxPool2D".
        """
        if stride is None:
            stride = pool_size

        super().__init__(stride, padding, name)

        self.pool_size = pool_size
        self.cache: Optional[dict] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Perform the forward pass of the MaxPool2D layer.

        Args:
            x (np.ndarray): The input to the layer.
            training (bool, optional): The flag indicating whether in training
                                       mode. Defaults to True.

        Returns:
            np.ndarray: The output of the layer.
        """
        N, C, H, W = x.shape
        pool_H, pool_W = self.pool_size, self.pool_size
        stride, padding = self.stride, self.padding

        # Compute output dimensions.
        out_H = 1 + (H + 2 * padding - pool_H) // stride
        out_W = 1 + (W + 2 * padding - pool_W) // stride

        # Convert input to columns.
        cols = image_to_column(x, pool_H, stride, padding)
        cols_reshaped = cols.reshape(C, pool_H * pool_W, N * out_H * out_W)

        # Max pooling.
        max_idx = np.argmax(cols_reshaped, axis=1)
        out = cols_reshaped[
            np.arange(C)[:, None], max_idx,
            np.arange(N * out_H * out_W)[None, :]
        ]
        out = out.reshape(N, C, out_H, out_W)

        # Cache for backward.
        self.cache = {
            "x": x,
            "cols": cols,
            "cols_reshaped": cols_reshaped,
            "max_idx": max_idx,
            "out_H": out_H,
            "out_W": out_W,
        }

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform the backward pass of the MaxPool2D layer.

        Args:
            dout (np.ndarray): The upstream gradient.

        Returns:
            np.ndarray: The downstream gradient.
        """
        x = self.cache["x"]
        cols_reshaped = self.cache["cols_reshaped"]
        max_idx = self.cache["max_idx"]
        N, C, H, W = x.shape
        pool_H, pool_W = self.pool_size, self.pool_size
        stride, padding = self.stride, self.padding
        out_H, out_W = self.cache["out_H"], self.cache["out_W"]

        # Flatten dout.
        dout_flat = dout.transpose(1, 0, 2, 3).reshape(C, -1)

        # Backprop into columns.
        dcols = np.zeros_like(cols_reshaped)

        np.put_along_axis(dcols, max_idx[:, None, :], dout_flat[:, None, :],
                          axis=1)

        # Reshape back to 2D column shape.
        dcols = dcols.reshape(C * pool_H * pool_W, N * out_H * out_W)

        # Convert columns back to image.
        dx = column_to_image(dcols, x.shape, pool_H, stride, padding)

        return dx

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
