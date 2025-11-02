import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    desired_classes: set,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the confusion matrix using matplotlib only.
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        desired_classes (set): Set of desired class labels to include in the confusion matrix.
        class_names (Optional[List[str]]): List of class names. If None, uses
                                           integers.
        save_path (Optional[str]): Path to save the plot. If None, the plot is
                                   shown.
    """
    # Create a mapping from desired class to index
    class_to_index = {cls: idx for idx, cls in enumerate(sorted(desired_classes))}
    num_classes = len(desired_classes)

    # Initialize the confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for true, pred in zip(y_true, y_pred):
        if true in desired_classes and pred in desired_classes:
            true_idx = class_to_index[true]
            pred_idx = class_to_index[pred]
            cm[true_idx][pred_idx] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    if class_names is None:
        class_names = [str(cls) for cls in sorted(desired_classes)]

    plt.xticks(np.arange(num_classes), class_names, rotation=45)
    plt.yticks(np.arange(num_classes), class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()




def plot_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    train_accs: List[float],
    val_accs: Optional[List[float]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the training and validation loss and accuracy curves.

    Args:
        train_losses (List[float]): List of training losses.
        val_losses (Optional[List[float]]): List of validation losses.
        train_accs (List[float]): List of training accuracies.
        val_accs (Optional[List[float]]): List of validation accuracies.
        save_path (Optional[str]): Path to save the plot. If None, the plot is
                                   shown.
    """
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    if val_accs is not None:
        plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def compute_saliency_map(model, input_sample, eps=1e-5, batch_size=1000):
    """
    Compute a saliency map for a single input sample using batched finite differences.

    Args:
        model: Model with a .forward() method.
        input_sample (np.ndarray): Single input, shape (C, H, W)
        eps (float): Small perturbation for finite differences
        batch_size (int): Number of pixels to process at a time

    Returns:
        np.ndarray: Saliency map, same shape as input_sample
    """
    input_sample = np.array(input_sample, dtype=float)

    # Forward pass for original input
    output = model.forward(input_sample[np.newaxis, ...]).reshape(-1)
    predicted_class = np.argmax(output)

    # Flatten input for pixel-wise perturbation
    flat_input = input_sample.flatten()
    D = flat_input.size

    # Initialize gradients array
    gradients = np.zeros(D)

    # Compute gradients in batches
    for start in range(0, D, batch_size):
        end = min(start + batch_size, D)
        batch_size_actual = end - start
        # Create perturbed inputs for this batch
        perturbed_inputs = np.tile(flat_input, (batch_size_actual, 1))
        # Perturb each pixel in the batch
        for i in range(batch_size_actual):
            perturbed_inputs[i, start + i] += eps
        # Reshape to original input shape for model
        perturbed_inputs = perturbed_inputs.reshape(batch_size_actual, *input_sample.shape)
        # Forward pass on perturbed inputs
        outputs_perturbed = model.forward(perturbed_inputs)
        # Compute gradients for this batch
        gradients[start:end] = (outputs_perturbed[:, predicted_class] - output[predicted_class]) / eps

    # Reshape gradients to input shape
    saliency_map = np.abs(gradients).reshape(input_sample.shape)
    return saliency_map


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
    plt.title("Saliency Map")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
