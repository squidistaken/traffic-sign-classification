import numpy as np
from typing import Tuple, Any
from nn.model import Model


def evaluate(model: Model, test_data: np.ndarray, test_labels: np.ndarray,
             loss_fn: Any, batch_size: int = 32, verbose: bool = True
             ) -> Tuple[float, float]:
    """
    Evaluate the model on a test dataset.

    Args:
        model (Model): The neural network model.
        test_data (np.ndarray): The test data.
        test_labels (np.ndarray): The test labels.
        loss_fn (Any): The loss function.
        batch_size (int, optional): The batch size for evaluation.
                                    Defaults to 32.
        verbose (bool, optional): The flag whether to print evaluation metrics.
                                  Defaults to True.

    Returns:
        Tuple[float, float]: The test loss and accuracy.
    """
    model.eval()

    test_loss = 0.0
    test_acc = 0.0

    # Iterate over the test dataset in batches.
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]

        # Forward pass.
        output = model.forward(batch_data)
        loss = loss_fn(output, batch_labels)

        # Calculate test metrics.
        test_loss += loss
        pred = np.argmax(output, axis=1)
        test_acc += np.sum(pred == batch_labels)

    # Calculate the average test loss and accuracy.
    test_loss /= len(test_data)
    test_acc /= len(test_data)

    if verbose:
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return test_loss, test_acc


def evaluate_corruption(model: Model, test_data: np.ndarray,
                        test_labels: np.ndarray, corruption_fn: Any,
                        loss_fn: Any, batch_size: int = 32,
                        verbose: bool = True) -> Tuple[float, float]:
    """
    Evaluate the model on a corrupted test dataset.

    Args:
        model (Model): The neural network model.
        test_data (np.ndarray): The test data.
        test_labels (np.ndarray): The test labels.
        corruption_fn (Any): The function to apply corruption to the test data.
        loss_fn (Any): The loss function.
        batch_size (int, optional): The batch size for evaluation.
                                    Defaults to 32.
        verbose (bool, optional): Whether to print evaluation metrics.
                                  Defaults to True.

    Returns:
        Tuple[float, float]: The corrupted test loss and accuracy.
    """
    model.eval()

    corrupted_test_loss = 0.0
    corrupted_test_acc = 0.0

    # Apply corruption to the test data.
    corrupted_test_data = corruption_fn(test_data)

    # Iterate over the corrupted test dataset in batches.
    for i in range(0, len(corrupted_test_data), batch_size):
        batch_data = corrupted_test_data[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]

        # Forward pass.
        output = model.forward(batch_data)
        loss = loss_fn(output, batch_labels)

        # Calculate corrupted test metrics.
        corrupted_test_loss += loss
        pred = np.argmax(output, axis=1)
        corrupted_test_acc += np.sum(pred == batch_labels)

    # Calculate the average corrupted test loss and accuracy.
    corrupted_test_loss /= len(corrupted_test_data)
    corrupted_test_acc /= len(corrupted_test_data)

    if verbose:
        print(f"Corrupted Test Loss: {corrupted_test_loss:.4f},"
              f"Corrupted Test Accuracy: {corrupted_test_acc:.4f}")

    return corrupted_test_loss, corrupted_test_acc
