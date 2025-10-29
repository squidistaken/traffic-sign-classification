import numpy as np


def check_data_shape(data: np.ndarray, expected_shape: tuple) -> bool:
    """
    Check if the data has the expected shape.

    Args:
        data (np.ndarray): The input data.
        expected_shape (tuple): The expected shape of the data.

    Returns:
        bool: The flag indicating if the data shape matches the expected shape.
    """
    return data.shape == expected_shape


def check_data_range(data: np.ndarray, min_val: float, max_val: float) -> bool:
    """Check if the data values are within a specified range.

    Args:
        data (np.ndarray): The input data.
        min_val (float): The minimum expected value.
        max_val (float): The maximum expected value.

    Returns:
        bool: The flag indicating if the data values are within the range.
    """
    return np.all(data >= min_val) and np.all(data <= max_val)


def check_model_performance(accuracy: float, threshold: float) -> bool:
    """Check if the model's accuracy meets a specific threshold.

    Args:
        accuracy (float): The model's accuracy.
        threshold (float): The minimum acceptable threshold.

    Returns:
        bool: The flag if the accuracy is above the threshold.
    """
    return accuracy >= threshold


def check_model_convergence(loss_values: list[float], patience: int = 5,
                            tol: float = 1e-4) -> bool:
    """Check if the model has converged based on the loss values.

    Args:
        loss_values (list[float]): The list of loss values over epochs.
        patience (int, optional): The number of epochs to wait before declaring
                                  convergence. Defaults to 5.
        tol (float, optional): The tolerance for considering the loss as
                               stable. Defaults to 1e-4.

    Returns:
        bool: The flag indicating if the loss has stablised within the
              tolerance for the specified patience.
    """
    if len(loss_values) < patience:
        return False

    recent_losses = loss_values[-patience:]

    return all(abs(loss - recent_losses[0]) < tol for loss in recent_losses)
