import numpy as np
import os
from logger import Logger
from typing import Tuple, List, Any
from nn.model import Model


def train(model: Model, train_data: np.ndarray, train_labels: np.ndarray,
          val_data: np.ndarray = None, val_labels: np.ndarray = None,
          loss_fn: Any = None, optimiser: Any = None, num_epochs: int = 10,
          batch_size: int = 32, checkpoint_dir: str = "checkpoints",
          log_dir: str = "logs"
          ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Train the model and log metrics.

    Args:
        model (Any): The neural network model.
        train_data (np.ndarray): The training data.
        train_labels (np.ndarray): The training labels.
        val_data (np.ndarray, optional): The validation data. Defaults to None.
        val_labels (np.ndarray, optional): The validation labels.
                                           Defaults to None.
        loss_fn (Any, optional): The loss function. Defaults to None.
        optimiser (Any, optional): The optimiser. Defaults to None.
        num_epochs (int, optional): The number of epochs to train for. Defaults
                                    to 10.
        batch_size (int, optional): The batch size for training. Defaults to
                                    32.
        checkpoint_dir (str, optional): The directory to save checkpoints.
                                        Defaults to "checkpoints".
        log_dir (str, optional): The directory to save logs. Defaults to
                                 "logs".

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]:
            The training loss and accuracy, and the validation loss and
            accuracy.
    """
    logger = Logger(log_dir=log_dir)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialise lists to store metrics.
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Training loop.
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0

        # Iterate over the training dataset in batches.
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]

            # Forward pass.
            output = model.forward(batch_data)
            loss = loss_fn(output, batch_labels)

            # Backward pass and optimise.
            model.backward(loss)
            optimiser.step()

            # Calculate training metrics
            epoch_train_loss += loss
            pred = np.argmax(output, axis=1)
            epoch_train_acc += np.sum(pred == batch_labels)

        # Calculate average training loss and accuracy for the epoch.
        epoch_train_loss /= len(train_data)
        epoch_train_acc /= len(train_data)
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Log training metrics
        logger.log_training_metrics(epoch, num_epochs, epoch_train_loss,
                                    epoch_train_acc)

        # Validation
        if val_data is not None and val_labels is not None:
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0

            # Iterate over the validation dataset in batches
            for i in range(0, len(val_data), batch_size):
                batch_data = val_data[i:i + batch_size]
                batch_labels = val_labels[i:i + batch_size]

                # Forward pass
                output = model.forward(batch_data)
                loss = loss_fn(output, batch_labels)

                # Calculate validation metrics
                epoch_val_loss += loss
                pred = np.argmax(output, axis=1)
                epoch_val_acc += np.sum(pred == batch_labels)

            # Calculate average validation loss and accuracy for the epoch
            epoch_val_loss /= len(val_data)
            epoch_val_acc /= len(val_data)
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)

            # Log validation metrics
            logger.log_validation_metrics(epoch, num_epochs, epoch_val_loss,
                                          epoch_val_acc)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir,
                                       f'model_epoch_{epoch + 1}.npz')
        np.savez(checkpoint_path, model_params=model.params())

    return train_losses, val_losses, train_accs, val_accs
