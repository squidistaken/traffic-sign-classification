from typing import Any, Tuple
import numpy as np


def evaluate(
    model: Any,
    data_loader: Any,
    loss_fn: Any,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Evaluate the model on a dataset using a DataLoader.

    This function computes the loss and accuracy of a model on a given dataset.

    Args:
        model (Any): The neural network model.
        data_loader (Any): Iterable yielding (batch_data, batch_labels).
        loss_fn (Any): The loss function (always returns (loss, grad)).
        verbose (bool, optional): Whether to print evaluation metrics. Defaults
                                  to True.

    Returns:
        Tuple[float, float]: Test loss and accuracy.
    """
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for batch_data, batch_labels in data_loader:
        output = model.forward(batch_data, False)
        loss, _ = loss_fn(output, batch_labels)  # always take first element
        total_loss += loss * len(batch_labels)
        pred = output.argmax(axis=1)
        total_acc += (pred == batch_labels).sum()
        total_samples += len(batch_labels)

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    if verbose:
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {avg_acc:.4f}")

    return avg_loss, avg_acc


def evaluate_corruption(
    model: Any,
    data_loader: Any,
    corruption_fn: Any,
    loss_fn: Any,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Evaluate the model on a corrupted dataset using a DataLoader.

    This function computes the loss and accuracy of a model on a given dataset
    after applying a corruption function to the input data.

    Args:
        model (Any): The neural network model.
        data_loader (Any): Iterable yielding (batch_data, batch_labels).
        corruption_fn (Callable): Function to apply corruption to a batch.
        loss_fn (Any): The loss function (always returns (loss, grad)).
        verbose (bool, optional): Whether to print evaluation metrics. Defaults
                                  to True.

    Returns:
        Tuple[float, float]: Corrupted test loss and accuracy.
    """
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for batch_data, batch_labels in data_loader:
        corrupted_batch_data = corruption_fn(batch_data)
        output = model.forward(corrupted_batch_data, False)
        loss, _ = loss_fn(output, batch_labels)  # always take first element
        total_loss += loss * len(batch_labels)
        pred = output.argmax(axis=1)
        total_acc += (pred == batch_labels).sum()
        total_samples += len(batch_labels)

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    if verbose:
        print(
            f"Corrupted Test Loss: {avg_loss:.4f}"
            + f"Corrupted Test Accuracy: {avg_acc:.4f}"
        )
    return avg_loss, avg_acc


def predict(model: Any, data_loader: Any) -> np.ndarray:
    """Generate predictions for a dataset using a custom DataLoader.

    This function generates predictions for a given dataset using the specified
    model.

    Args:
        model (Any): The neural network model.
        data_loader (Any): Iterable yielding batches of data.

    Returns:
        np.ndarray: Array of predicted labels.
    """
    predictions_list = []

    for batch_data, _ in data_loader:
        output = model.forward(batch_data, False)
        pred = output.argmax(axis=1)
        predictions_list.append(pred)

    return np.concatenate(predictions_list)
