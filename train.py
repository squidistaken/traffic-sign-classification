import numpy as np
import os
from logger import Logger
from typing import Tuple, List, Any


def train(
    model: Any,
    train_loader: Any,
    val_loader: Any = None,
    loss_fn: Any = None,
    optimiser: Any = None,
    num_epochs: int = 10,
    batch_size: int = 32,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train the model and log metrics.
    Returns:
        Tuple of training losses, validation losses, training accuracies, validation accuracies.
    """
    logger = Logger(log_dir=log_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    logger.log_debug("Training started.")

    for epoch in range(num_epochs):
        logger.log_debug(f"Starting epoch {epoch + 1}/{num_epochs}.")
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        n_train_samples = 0
        # Shuffle training data
        for b_data, b_labels in train_loader:
            logger.log_debug(
                f"Processing training batch with {b_data.shape[0]} samples."
            )
            n_train_samples += b_data.shape[0]
            # Reset optimiser gradients
            optimiser.zero_grad()
            # Forward pass
            output = model.forward(b_data, training=True)
            # Loss and gradient
            loss, grad_output = loss_fn(output, b_labels)
            logger.log_debug(f"Training batch loss: {loss}")
            # Backward pass
            model.backward(grad_output)
            optimiser.step()
            # Accumulate metrics
            epoch_train_loss += loss * b_data.shape[0]
            preds = np.argmax(output, axis=1)
            epoch_train_acc += np.sum(preds == b_labels)
            logger.log_debug(
                f"Training batch accuracy: {np.sum(preds == b_labels) / b_data.shape[0]}"
            )
        # Average metrics
        epoch_train_loss /= n_train_samples
        epoch_train_acc /= n_train_samples
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        logger.log_training_metrics(
            epoch, num_epochs, epoch_train_loss, epoch_train_acc
        )
        # Validation loop
        if val_loader:
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0
            n_val_samples = 0
            for b_data, b_labels in val_loader:
                logger.log_debug(
                    f"Processing validation batch with {b_data.shape[0]} samples."
                )
                n_val_samples += b_data.shape[0]
                output = model.forward(b_data, training=False)
                loss, _ = loss_fn(output, b_labels)
                logger.log_debug(f"Validation batch loss: {loss}")
                epoch_val_loss += loss * b_data.shape[0]
                preds = np.argmax(output, axis=1)
                epoch_val_acc += np.sum(preds == b_labels)
                logger.log_debug(
                    f"Validation batch accuracy: {np.sum(preds == b_labels) / b_data.shape[0]}"
                )
            epoch_val_loss /= n_val_samples
            epoch_val_acc /= n_val_samples
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)
            logger.log_validation_metrics(
                epoch, num_epochs, epoch_val_loss, epoch_val_acc
            )
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.npz")
        np.savez(checkpoint_path, model_params=model.params())
        logger.log_debug(f"Checkpoint saved to {checkpoint_path}")
    logger.log_debug("Training completed.")
    return train_losses, val_losses, train_accs, val_accs
