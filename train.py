import numpy as np
import os
from logger import Logger
from typing import Tuple, List, Any

def train(model: Any,
          train_data: np.ndarray,
          train_labels: np.ndarray,
          val_data: np.ndarray = None,
          val_labels: np.ndarray = None,
          loss_fn: Any = None,
          optimiser: Any = None,
          num_epochs: int = 10,
          batch_size: int = 32,
          checkpoint_dir: str = "checkpoints",
          log_dir: str = "logs"
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

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0

        # Shuffle training data
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)
        train_data_shuffled = train_data[indices]
        train_labels_shuffled = train_labels[indices]

        # Training loop
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data_shuffled[i:i + batch_size]
            batch_labels = train_labels_shuffled[i:i + batch_size]

            # Forward pass
            output = model.forward(batch_data, training=True)

            # Compute loss and gradient
            loss, grad_output = loss_fn(output, batch_labels)

            # Backward pass
            model.backward(grad_output)
            optimiser.step()

            # Accumulate metrics
            epoch_train_loss += loss
            preds = np.argmax(output, axis=1)
            epoch_train_acc += np.sum(preds == batch_labels)

        # Average metrics
        epoch_train_loss /= len(train_data)
        epoch_train_acc /= len(train_data)
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        logger.log_training_metrics(epoch, num_epochs, epoch_train_loss, epoch_train_acc)

        # Validation loop
        if val_data is not None and val_labels is not None:
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0

            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i + batch_size]
                batch_labels = val_labels[i:i + batch_size]

                output = model.forward(batch_data, training=False)
                loss, _ = loss_fn(output, batch_labels)

                epoch_val_loss += loss
                preds = np.argmax(output, axis=1)
                epoch_val_acc += np.sum(preds == batch_labels)

            epoch_val_loss /= len(val_data)
            epoch_val_acc /= len(val_data)
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)

            logger.log_validation_metrics(epoch, num_epochs, epoch_val_loss, epoch_val_acc)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.npz')
        np.savez(checkpoint_path, model_params=model.params())

    return train_losses, val_losses, train_accs, val_accs
