import numpy as np
from typing import Tuple


def image_to_column(
    x: np.ndarray, kernel_size: int, stride: int, padding: int = 0
) -> np.ndarray:
    """Convert image to columns for efficient convolution.

    Args:
        x (np.ndarray): The input image in NCHW format.
        kernel_size (int): The size of the convolution kernel.
        stride (int): The stride of the convolution.
        padding (int, optional): Zero-padding added to both sides of the
                                 input. Defaults to 0.

    Returns:
        np.ndarray: The input image converted to columns.
    """
    N, C, H, W = x.shape

    # Compute the output dimensions.
    out_H = 1 + (H + 2 * padding - kernel_size) // stride
    out_W = 1 + (W + 2 * padding - kernel_size) // stride

    # Pad the input, if necessary.
    if padding > 0:
        x_padded = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
        )
    else:
        x_padded = x

    # Initialize the output
    cols = np.zeros((C * kernel_size * kernel_size, out_H * out_W * N))

    # Convert image to columns
    # For each sample in the batch.
    for n in range(N):
        # For each output row.
        for i in range(out_H):
            # For each output column.
            for j in range(out_W):
                # Compute the start and end indices for the current patch.
                h_start = i * stride
                h_end = h_start + kernel_size
                w_start = j * stride
                w_end = w_start + kernel_size

                # Extract the current patch and flatten it
                patch = x_padded[n, :, h_start:h_end, w_start:w_end]
                cols[:, n * out_H * out_W + i * out_W + j] = patch.flatten()

    return cols


def column_to_image(
    dx_cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    kernel_size: int,
    stride: int,
    padding: int = 0,
) -> np.ndarray:
    """
    Convert columns to image after convolution.

    Args:
        dx_cols (np.ndarray): The gradient with respect to the input in column
                              format.
        x_shape (Tuple[int, int, int, int]): The shape of the original input.
        kernel_size (int): The size of the convolution kernel.
        stride (int): The stride of the convolution.
        padding (int, optional): Zero-padding added to both sides of the
                                 input. Defaults to 0.

    Returns:
        np.ndarray: The gradient with respect to the input in image format.
    """
    N, C, H, W = x_shape

    # Compute the output dimensions.
    out_H = 1 + (H + 2 * padding - kernel_size) // stride
    out_W = 1 + (W + 2 * padding - kernel_size) // stride

    # Initialize the output.
    dx = np.zeros((N, C, H + 2 * padding, W + 2 * padding))

    # Convert the columns back to image.
    # For each sample in the batch.
    for n in range(N):
        # For each output row.
        for i in range(out_H):
            # For each output column.
            for j in range(out_W):
                # Compute the start and end indices for the current patch.
                h_start = i * stride
                h_end = h_start + kernel_size
                w_start = j * stride
                w_end = w_start + kernel_size

                # Extract the current patch and reshape it
                patch = dx_cols[:, n * out_H * out_W + i * out_W + j]
                dx[n, :, h_start:h_end, w_start:w_end] += patch.reshape(
                    C, kernel_size, kernel_size
                )

    # Remove the padding, if necessary.
    if padding > 0:
        dx = dx[:, :, padding:-padding, padding:-padding]

    return dx


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert labels to one-hot encoded format.

    Args:
        labels (np.ndarray): The input labels.
        num_classes (int): The number of classes.

    Returns:
        np.ndarray: The one-hot encoded labels.
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1

    return one_hot


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the accuracy of predictions.

    Args:
        predictions (np.ndarray): The predicted labels.
        labels (np.ndarray): The true labels.

    Returns:
        float: The accuracy of the predictions.
    """
    correct = np.sum(predictions == labels)
    total = labels.shape[0]
    accuracy = correct / total

    return accuracy
