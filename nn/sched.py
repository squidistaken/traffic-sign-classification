# StepLR, Cosine, Warmup
from math import pi, cos


# region Step LR Scheduler
class StepLRScheduler:
    def __init__(self, base_lr: float, step_size: int,
                 gamma: float = 0.1) -> None:
        """ Step learning rate scheduler.

        Args:
            base_lr (float): Learning rate at the start of training.
            step_size (int): Amount of epochs between learning rate decay.
            gamma (float): Multiplicative factor of decay. Defaults to 0.1.
        """
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, curr_epoch: int) -> float:
        """ Returns decayed learning rate for the current epoch.

        Args:
            curr_epoch (int): Number of current epoch.

        Returns:
            float: Decayed learning rate.
        """
        # For every decay step, multiply base lr by gamma
        amount_of_steps = curr_epoch // self.step_size
        return self.base_lr * (self.gamma ** amount_of_steps)

    def __call__(self, curr_epoch: int) -> float:
        """Callable class method for getting learning rate.
        Shortcut for class method get_lr.

        Args:
            curr_epoch (int): Number of current epoch.

        Returns:
            float: Decayed learning rate.
        """
        return self.get_lr(curr_epoch)
# endregion

# region Cosine LR Scheduler
class CosineLRScheduler:
    def __init__(self, base_lr: float, lowest_lr: float,
                 total_epochs: int) -> None:
        """ Initialization of cosine annealer class. This
        class slowly reduces the learning rate using cosine curves.

        Args:
            base_lr (float): Learning rate at the start of training.
            lowest_lr (float): Lowest value the learning rate can become.
            total_epochs (int): Total amount of training epochs.
        """
        self.base_lr = base_lr
        self.lowest_lr = lowest_lr
        self.total_epochs = total_epochs

    def get_lr(self, curr_epoch: int) -> float:
        """ Returns cosine annealed learning rate for current epoch.

        Args:
            curr_epoch (int): Number of current epoch.

        Returns:
            float: Annealed learning rate.
        """
        lr = (
            self.lowest_lr + 0.5 * (self.base_lr - self.lowest_lr) *
            (1 + cos(pi * curr_epoch / self.total_epochs))
        )
        return lr

    def __call__(self, curr_epoch: int) -> float:
        """Callable class method for getting learning rate.
        Shortcut for class method get_lr.

        Args:
            curr_epoch (int): Number of current epoch.

        Returns:
            float: Annealed learning rate.
        """
        return self.get_lr(curr_epoch)
# endregion
