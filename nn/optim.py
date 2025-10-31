import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from .layers.base_layers import Layer


# region Abstract Optimizer
class Optimizer(ABC):
    """Abstract Base Class for Optimisers"""

    def __init__(
        self,
        raw_params: list[Tuple[Layer, str, np.ndarray]] | list[Dict],
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialise the Optimiser class.

        Args:
            raw_params (list[Tuple[Layer, str, np.ndarray]] | list[Dict]): The
                - iterable of triples (layer, name, array); or
                - list of parameter groups.
            lr (float, optional): The learning rate. Defaults to 0.001.
            weight_decay (float, optional): The weight decay (L2 penalty).
                                            Defaults to 0.0.
        """
        self.param_groups = self._make_param_groups(raw_params, lr, weight_decay)
        # Per-parameter state dictionaries
        self.state = {}

    @abstractmethod
    def step(self) -> None:
        """Perform a single optimisation step."""
        pass

    def zero_grad(self) -> None:
        """Set all parameter gradients to zero."""
        for group in self.param_groups:
            for layer, name, param in group["params"]:
                grad = getattr(layer, "grad_" + name)
                grad.fill(0.0)

    def _make_param_groups(
        self,
        raw_params: list[Tuple[Layer, str, np.ndarray]] | list[Dict],
        lr: float,
        weight_decay: float,
    ) -> list[Dict[str, Any]]:
        """Turn the raw parameters inputs into a standardised structure.

        Args:
            raw_params (list[Tuple[Layer, str, np.ndarray]] | list[Dict]):
                The
                - iterable of triples (layer, name, array); or
                - list of parameter groups.
            lr (float): The learning rate.
            weight_decay (float): The weight decay (L2 penalty).

        Raises:
            ValueError: If the parameters are not an iterable of triples or
                        list of parameter groups.
            ValueError: If the parameter group is not a list.

        Returns:
            list[Dict[str, Any]]: The list of parameter groups in a
                                  standaradised structure.
        """
        param_groups = []

        # Handle both possible cases of argument params,
        # specified in docstring.
        if isinstance(raw_params, list):
            # Case 1: params is a list of (layer, name, array) tuples.
            if all(isinstance(p, tuple) and len(p) == 3 for p in raw_params):
                param_groups.append(
                    {
                        "params": raw_params,
                        "lr": lr,
                        "weight_decay": weight_decay,
                    }
                )
            # Case 2: params is already a list of a dictionary.
            elif all(isinstance(p, dict) for p in raw_params):
                for group in raw_params:
                    group_copy = group.copy()

                    # Ensure all groups have correct keys by setting deafults.
                    group_copy.setdefault("lr", lr)
                    group_copy.setdefault("weight_decay", weight_decay)
                    param_groups.append(group_copy)
            else:
                raise ValueError(
                    (
                        "Parameters must be a list of an iterable of triples "
                        "(layer, name, array) tuples "
                        "OR a list of parameter group dictionaries."
                    )
                )
        else:
            raise ValueError("Parameters must be a list.")

        return param_groups


# endregion


# region SGD Optimizer
class SGD(Optimizer):
    """The Stochastic Gradient Descent optimiser class."""

    def step(self) -> None:
        """Perform a single optimisation step."""
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for layer, name, parameters in group["params"]:
                grad = getattr(layer, "grad_" + name)

                if weight_decay != 0:
                    upd = grad + weight_decay * parameters
                else:
                    # Skip unnecessary computations if there's no weight decay.
                    upd = grad

                # theta = theta - lr * (grad + weight_decay * theta).
                parameters -= lr * upd


# endregion


# region Adam Optimizer
class Adam(Optimizer):
    """The Adam optimiser class."""

    def step(self) -> None:
        """Perform a single optimisation step."""
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            # Get the Adam specific parameters from group, else get the
            # default.

            beta1 = group.get("beta1", 0.9)
            beta2 = group.get("beta2", 0.999)
            epsilon = group.get("epsilon", 1e-8)

            for layer, name, parameters in group["params"]:
                grad = getattr(layer, "grad_" + name)

                # Apply weight decay if it is not zero.
                if weight_decay != 0:
                    grad = grad + weight_decay * parameters

                # Initialize per-parameter state if not there
                if (layer, name) not in self.state:
                    self.state[(layer, name)] = {
                        "m": np.zeros_like(parameters),
                        "v": np.zeros_like(parameters),
                        "t": 0,
                    }

                state = self.state[(layer, name)]
                m, v, t = state["m"], state["v"], state["t"]

                # Update biased moment estimates
                t += 1
                state["t"] = t
                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * (grad**2)

                # Compute bias-corrected moment estimates
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # Parameter update
                upd = m_hat / (np.sqrt(v_hat) + epsilon)
                parameters -= lr * upd


# endregion


# region Momentum Optimizer
class Momentum(Optimizer):
    """The Momentum optimiser class."""

    def step(self) -> None:
        """Perform a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            # Get the parameters specific to Momentum or the default.
            momentum = group.get("momentum", 0.9)

            for layer, name, parameters in group["params"]:
                grad = getattr(layer, "grad_" + name)

                # Apply weight decay if it is not zero.
                if weight_decay != 0:
                    grad = grad + weight_decay * parameters

                # Initialize per-parameter state if not there
                if (layer, name) not in self.state:
                    self.state[(layer, name)] = {"v": np.zeros_like(parameters)}

                state = self.state[(layer, name)]
                v = state["v"]

                # Update the velocity.
                v[:] = momentum * v + grad

                # Update parameters.
                parameters -= lr * v


# endregion
