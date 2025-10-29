import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List


class LRScheduler(ABC):
    """Abstract Base Class for Learning Rate Schedulers"""
    @abstractmethod
    def get_lr(self, curr_epoch: int) -> float:
        """
        Get the learning rate for the current epoch.

        Args:
            curr_epoch (int): The current epoch number.

        Returns:
            float: The learning rate.
        """
        pass

    def __call__(self, curr_epoch: int) -> float:
        return self.get_lr(curr_epoch)

    def get_lrs(self, epochs: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Get the learning rates for the given epochs.

        Args:
            epoch (Union[List[int], np.ndarray]): The list or array of epoch
                                                  numbers.

        Returns:
            np.ndarray: The array of learning rates.
        """
        return np.array([self.get_lr(epoch) for epoch in epochs])


# region Step LR Scheduler
class StepLRScheduler(LRScheduler):
    """The Step Learning Rate Scheduler class."""
    def __init__(self, base_lr: float, step_size: int, gamma: float = 0.1
                 ) -> None:
        """
        Initialise the step learning rate scheduler class.

        Args:
            base_lr (float): The learning rate at the start of training.
            step_size (int): The amount of epochs between learning rate decay.
            gamma (float, optional): The multiplicative factor of decay.
                                     Defaults to 0.1.
        """
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, curr_epoch: int) -> float:
        """
        Get the learning rate for the current epoch.

        Args:
            curr_epoch (int): The current epoch number.

        Returns:
            float: The learning rate.
        """
        # For every decay step, multiply base lr by gamma
        amount_of_steps = curr_epoch // self.step_size
        lr = self.base_lr * (self.gamma ** amount_of_steps)

        return lr
# endregion


# region Cosine LR Scheduler
class CosineLRScheduler(LRScheduler):
    """
    The Cosine Learning Rate/Annealer class, which slowly reduces the learning
    rate using cosine curves.
    """
    def __init__(self, base_lr: float, lowest_lr: float, total_epochs: int
                 ) -> None:
        """
        Initialise the cosine annealer class.

        Args:
            base_lr (float): The learning rate at the start of training.
            lowest_lr (float): The minimum value the learning rate can reach.
            total_epochs (int): The total amount of training epochs.
        """
        self.base_lr = base_lr
        self.lowest_lr = lowest_lr
        self.total_epochs = total_epochs

    def get_lr(self, curr_epoch: int) -> float:
        """Get the learning rate for the current epoch.

        Args:
            curr_epoch (int): The current epoch number.

        Returns:
            float: The learning rate.
        """
        lr = (
            self.lowest_lr + 0.5 * (self.base_lr - self.lowest_lr) *
            (1 + np.cos(np.pi * curr_epoch / self.total_epochs))
        )

        return lr
# endregion


# region Warmup LR Scheduler
class WarmupLRScheduler(LRScheduler):
    """The Warmup Learning Rate Scheduler class."""
    def __init__(self, base_lr: float, warmup_epochs: int,
                 post_sched: callable | None = None) -> None:
        """Initialise the warmup learning rate scheduler class.

        Args:
            base_lr (float): The learning rate at the start of training.
            warmup_epochs (int): The amount of epochs it takes for the learning
                                 rate to increase to the base learning rate.
            post_sched (callable | None, optional):
                The additional scheduler for getting the learning rate after
                the warmup finishes. Defaults to None.
        """
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.scheduler = post_sched

    def get_lr(self, curr_epoch: int) -> float:
        """Get the learning rate for the current epoch.

        Args:
            curr_epoch (int): The current epoch number.

        Returns:
            float: The learning rate.
        """
        if curr_epoch < self.warmup_epochs:
            lr = self.base_lr * (curr_epoch / self.warmup_epochs)
        else:
            if self.scheduler:
                lr = self.scheduler(curr_epoch - self.warmup_epochs)
            else:
                lr = self.base_lr

        return lr
# endregion
