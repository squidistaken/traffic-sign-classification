import logging
import os


class Logger:
    """Logger class for managing logging operations.

    This class provides functionality for logging messages to both a file and
    the console, with configurable log levels and formats.
    """
    # Class variable to keep track of the number of training sessions.
    session_count = 0

    def __init__(self, log_dir: str = "logs", log_file: str = None):
        """Initialise the logger with the specified directory and file.

        This method sets up the logger with the specified directory and file
        name. If no file name is provided, a unique name is generated based on
        the session count.

        Args:
            log_dir (str, optional): Directory to save logs. Defaults to
                                     "logs".
            log_file (str, optional): Name of the log file. Defaults to None.
        """
        self.log_dir = log_dir
        Logger.session_count += 1  # Increment the session count.
        self.log_file = (log_file if log_file
                         else f"train_{Logger.session_count}.log")
        self.logger = logging.getLogger()

        self._setup_logger()

    def _setup_logger(self) -> None:
        """Set up the logger with the specified configuration.

        This method configures the logger with file and console handlers, and
        sets the logging level and format.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, self.log_file)

        # Create a file handler.
        file_handler = logging.FileHandler(log_path)

        file_handler.setLevel(logging.DEBUG)
        # Create a console handler.
        console_handler = logging.StreamHandler()

        console_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the handlers.
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger.
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

    def log_training_metrics(
        self, epoch: int, num_epochs: int, train_loss: float, train_acc: float
    ) -> None:
        """Log the training metrics.

        This method logs the training loss and accuracy for the specified
        epoch.

        Args:
            epoch (int): The current epoch.
            num_epochs (int): The total number of epochs.
            train_loss (float): The training loss.
            train_acc (float): The training accuracy.
        """
        self.logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_acc:.4f}"
        )

    def log_validation_metrics(
        self, epoch: int, num_epochs: int, val_loss: float, val_acc: float
    ) -> None:
        """Log the validation metrics.

        This method logs the validation loss and accuracy for the specified
        epoch.

        Args:
            epoch (int): The current epoch.
            num_epochs (int): The total number of epochs.
            val_loss (float): The validation loss.
            val_acc (float): The validation accuracy.
        """
        self.logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
        )

    def log_debug(self, message: str) -> None:
        """Log a debug message.

        This method logs a debug message to the configured logger.

        Args:
            message (str): The debug message to log.
        """
        self.logger.debug(message)
