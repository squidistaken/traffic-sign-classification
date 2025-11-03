import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Union, List


class LRScheduler(ABC):
    """Abstract Base Class for Learning Rate Schedulers.

    This class serves as the base class for all learning rate schedulers,
    providing the interface and common functionality required for adjusting the
    learning rate during training.
    """
    @abstractmethod
    def get_lr(self, curr_epoch: int) -> float:
        """Get the learning rate for the current epoch.

        This method computes the learning rate for the specified epoch
        according to the scheduler's algorithm.

        Args:
            curr_epoch (int): The current epoch number.

        Returns:
            float: The learning rate.
        """
        pass

    def __call__(self, curr_epoch: int) -> float:
        """Get the learning rate for the current epoch.

        This method is a convenience wrapper around `get_lr`.

        Args:
            curr_epoch (int): The current epoch number.

        Returns:
            float: The learning rate.
        """
        return self.get_lr(curr_epoch)

    def get_lrs(self, epochs: Union[List[int], np.ndarray]) -> np.ndarray:
        """Get the learning rates for the given epochs.

        This method computes the learning rates for a list of epochs according
        to the scheduler's algorithm.

        Args:
            epochs (Union[List[int], np.ndarray]): The list or array of epoch
                                                    numbers.

        Returns:
            np.ndarray: The array of learning rates.
        """
        return np.array([self.get_lr(epoch) for epoch in epochs])


# region Step LR Scheduler
class StepLRScheduler(LRScheduler):
    """The Step Learning Rate Scheduler class.

    This class implements a learning rate scheduler that decays the learning
    rate by a multiplicative factor every specified number of epochs.
    """
    def __init__(self, base_lr: float, step_size: int, gamma: float = 0.1
                 ) -> None:
        """Initialise the step learning rate scheduler class.

        This method initialises the scheduler with the base learning rate, the
        step size, and the decay factor.

        Args:
            base_lr (float): The learning rate at the start of training.
            step_size (int): The number of epochs between learning rate decays.
            gamma (float, optional): The multiplicative factor of decay.
                                     Defaults to 0.1.
        """
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, curr_epoch: int) -> float:
        """Get the learning rate for the current epoch.

        This method computes the learning rate for the specified epoch using
        the step decay schedule.

        Args:
            curr_epoch (int): The current epoch number.

        Returns:
            float: The learning rate.
        """
        # For every decay step, multiply base lr by gamma.
        amount_of_steps = curr_epoch // self.step_size
        lr = self.base_lr * (self.gamma**amount_of_steps)

        return lr

# endregion


# region Cosine LR Scheduler
class CosineLRScheduler(LRScheduler):
    """The Cosine Learning Rate Scheduler class.

    This class implements a learning rate scheduler that adjusts the learning
    rate using a cosine annealing schedule, gradually reducing it from the base
    learning rate to the lowest specified rate over the course of training.
    """

    def __init__(self, base_lr: float, lowest_lr: float, total_epochs: int
                 ) -> None:
        """Initialise the cosine learning rate scheduler class.

        This method initialises the scheduler with the base learning rate, the
        minimum learning rate, and the total number of training epochs.

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

        This method computes the learning rate for the specified epoch using
        the cosine annealing schedule.

        Args:
            curr_epoch (int): The current epoch number.

        Returns:
            float: The learning rate.
        """
        lr = self.lowest_lr + 0.5 * (self.base_lr - self.lowest_lr) * (
            1 + np.cos(np.pi * curr_epoch / self.total_epochs)
        )

        return lr
# endregion


# region Warmup LR Scheduler
class WarmupLRScheduler(LRScheduler):
    """The Warmup Learning Rate Scheduler class.

    This class implements a learning rate scheduler that gradually increases
    the learning rate from zero to the base learning rate over a specified
    number of epochs, optionally followed by another scheduler.
    """

    def __init__(self, base_lr: float, warmup_epochs: int,
                 post_sched: Callable | None = None) -> None:
        """Initialise the warmup learning rate scheduler class.

        This method initialises the scheduler with the base learning rate, the
        number of warmup epochs, and an optional post-warmup scheduler.

        Args:
            base_lr (float): The learning rate at the start of training.
            warmup_epochs (int): The number of epochs for the warmup phase.
            post_sched (callable | None, optional): The scheduler to use after
                                                    warmup. Defaults to None.
        """
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.scheduler = post_sched

    def get_lr(self, curr_epoch: int) -> float:
        """Get the learning rate for the current epoch.

        This method computes the learning rate for the specified epoch using
        the warmup schedule, optionally followed by another scheduler.

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
