import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import seaborn as sns


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (Optional[List[str]]): List of class names. If None, uses
                                           integers.
        save_path (Optional[str]): Path to save the plot. If None, the plot is
                                   shown.
    """
    # Determine the number of classes.
    num_classes = len(np.unique(y_true))

    # Initialise the confusion matrix with zeros.
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # Fill the confusion matrix.
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    # Plot the confusion matrix.
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the training and validation loss and accuracy curves.

    Args:
        train_losses (List[float]): List of training losses.
        val_losses (List[float]): List of validation losses.
        train_accs (List[float]): List of training accuracies.
        val_accs (List[float]): List of validation accuracies.
        save_path (Optional[str]): Path to save the plot. If None, the plot is
                                   shown.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")

    if val_losses:
        plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")

    if val_accs:
        plt.plot(val_accs, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_saliency_map(
    saliency_map: np.ndarray, save_path: Optional[str] = None
) -> None:
    """
    Plot a saliency map.

    Args:
        saliency_map (np.ndarray): The saliency map.
        save_path (Optional[str]): The path to save the plot. If None, the plot
                                   is shown.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(saliency_map, cmap="hot")
    plt.colorbar()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
