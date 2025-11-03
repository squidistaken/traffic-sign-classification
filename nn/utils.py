import csv
import os
import numpy as np
from typing import Tuple, Any, Set, List


def image_to_column(x: np.ndarray, kernel_size: int, stride: int,
                    padding: int = 0) -> np.ndarray:
    """Convert an image to columns for efficient convolution.

    This function transforms a batch of images into a matrix of columns, where
    each column represents a flattened patch from the image, facilitating
    efficient convolution operations.

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
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant"
        )
    else:
        x_padded = x

    # Initialise the output.
    cols = np.zeros((C * kernel_size * kernel_size, out_H * out_W * N))

    # Convert image to columns.
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
                # Extract the current patch and flatten it.
                patch = x_padded[n, :, h_start:h_end, w_start:w_end]
                cols[:, n * out_H * out_W + i * out_W + j] = patch.flatten()
    return cols


def column_to_image(dx_cols: np.ndarray, x_shape: Tuple[int, int, int, int],
                    kernel_size: int, stride: int, padding: int = 0
                    ) -> np.ndarray:
    """Convert columns to an image after convolution.

    This function reconstructs an image from a matrix of columns, where each
    column represents a flattened patch from the original image, converting it
    back to the original image format.

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

    # Initialise the output.
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

                # Extract the current patch and reshape it.
                patch = dx_cols[:, n * out_H * out_W + i * out_W + j]
                dx[n, :, h_start:h_end, w_start:w_end] += patch.reshape(
                    C, kernel_size, kernel_size
                )

    # Remove the padding, if necessary.
    if padding > 0:
        dx = dx[:, :, padding:-padding, padding:-padding]

    return dx


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert labels to one-hot encoded format.

    This function transforms categorical labels into a binary matrix where each
    row represents a sample, and each column represents a class. A value of 1
    indicates the presence of the class.

    Args:
        labels (np.ndarray): The input labels.
        num_classes (int): The number of classes.

    Returns:
        np.ndarray: The one-hot encoded labels.
    """
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1

    return one_hot


def export_encoded_labels(dataset: Any, chosen_classes: Set[int],
                          clipped_indices: List[int],
                          output_name: str = "filtered_labels.csv"
                          ) -> str:
    """Export a filtered and encoded GTSRB labels file matching the original
    format.

    This function exports a CSV file containing filtered and re-encoded labels
    for the specified dataset, preserving the original format.

    Args:
        dataset (Any): The dataset containing the labels and metadata.
        chosen_classes (Set[int]): The selected class labels.
        clipped_indices (List[int]): The dataset indices to include.
        output_name (str, optional): Name of the output CSV file. Defaults to
                                     'filtered_labels.csv'.

    Returns:
        str: Path to the exported CSV file.
    """
    labels = dataset.labels_data
    root = dataset.root
    labels_path = os.path.join(root, dataset.labels)

    # Load the original CSV to preserve all fields.
    with open(labels_path, "r") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        full_rows = [row for row in reader]

    # Map classes to new indices (0..N-1).
    label_to_index = {label: index for index,
                      label in enumerate(chosen_classes)}
    mapped_labels = np.array([label_to_index[labels[i][1]]
                              for i in clipped_indices])

    # Build filtered rows with new encoded ClassId.
    filtered_rows = []

    for idx, mapped_label in zip(clipped_indices, mapped_labels):
        row = full_rows[idx]
        row[-1] = str(mapped_label)
        filtered_rows.append(row)

    # Write to new CSV with tje exact same columns.
    output_csv = os.path.join(root, output_name)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        writer.writerow(header)
        writer.writerows(filtered_rows)

    return output_csv


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute the accuracy of predictions.

    This function calculates the proportion of correct predictions out of the
    total number of predictions made.

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
