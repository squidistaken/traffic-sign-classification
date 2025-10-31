import logging
import os


class Logger:
    """Logger class."""

    def __init__(self, log_dir: str = "logs", log_file: str = "training.log"):
        """
        Initialise the logger.

        Args:
            log_dir (str): Directory to save logs.
            log_file (str): Name of the log file.
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.logger = logging.getLogger()

        self._setup_logger()

    def _setup_logger(self) -> None:
        """Set up the logger with the specified configuration."""
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, self.log_file)
        logging.basicConfig(filename=log_path, level=logging.INFO)

    def log_training_metrics(
        self, epoch: int, num_epochs: int, train_loss: float, train_acc: float
    ) -> None:
        """
        Log the training metrics.

        Args:
            epoch (int): The current epoch.
            num_epochs (int): The total number of epochs.
            train_loss (float): The training loss.
            train_acc (float): The training accuracy.
        """
        self.logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f},"
            f"Train Accuracy: {train_acc:.4f}"
        )

    def log_validation_metrics(
        self, epoch: int, num_epochs: int, val_loss: float, val_acc: float
    ) -> None:
        """
        Log the validation metrics.

        Args:
            epoch (int): The current epoch.
            num_epochs (int): The total number of epochs.
            val_loss (float): The validation loss.
            val_acc (float): The validation accuracy.
        """
        self.logger.info(
            f"Epoch {epoch + 1}/{num_epochs},"
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
        )
