# SGD, Momentum, Adam
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from .layers.base_layer import Layer


# region Abstract Optimizer
class Optimizer(ABC):
    """Abstract base class for all optimizers."""

    def __init__(
        self,
        raw_params: list[Tuple[Layer, str, np.ndarray]] | list[Dict],
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialization of base Optimizer class.

        Args:
            raw_params (list[Tuple[Layer, str, np.ndarray]] | list[Dict]):
                Either:
                - An iterable of triples (layer, name, array) where layer is a
                Layer instance, name is the parameter name (str),
                and array is the parameter tensor (np.ndarray),
                - OR a list of 'param groups': dicts with parameters and
                per-group hyperparameters
            lr (float): Learning rate. Defaults to 0.001.
            weight_decay (float): Weight decay (L2 penalty).
                                  Defaults to 0.0.
        """
        self.param_groups = self._make_param_groups(
            raw_params, lr, weight_decay
        )
        self.state = {}  # per-parameter state dictionaries

    @abstractmethod
    def step(self) -> None:
        """Abstract method for single optimization step."""
        pass

    def zero_grad(self) -> None:
        """Sets all parameter gradients to zero."""
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
        """
        Turns the two types of raw_params inputs
        into a standardized structure.

        Args:
            raw_params (list[Tuple[Layer, str, np.ndarray]] | list[Dict]):
                Either:
                - An iterable of triples (layer, name, array) where layer is a
                Layer instance, name is the parameter name (str),
                and array is the parameter tensor (np.ndarray),
                - OR a list of 'param groups': dicts with parameters and
                per-group hyperparameters
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 penalty).

        Returns:
            list[Dict[str, Any]]: List of parameter groups in
                                  a standardized structure.
        """

        param_groups = []

        # Handle both possible cases of argument params,
        # specified in docstring.
        if isinstance(raw_params, list):
            # Case 1: params is a list of (layer, name, array) tuples.
            if all(isinstance(p, tuple) and len(p) == 3 for p in raw_params):
                param_groups.append({
                    "params": raw_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                })
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
                        "Params must be a list of an iterable of triples "
                        "(layer, name, array) tuples "
                        "OR a list of parameter group dictionaries."
                    )
                )
        else:
            raise ValueError("Params must be a list")

        return param_groups
# endregion


# region SGD Optimizer
class SGD(Optimizer):
    """Implements stochastic gradient descent optimizer."""

    def step(self) -> None:
        """Performs a single optimization step."""

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

                # theta = theta - lr * (grad + weight_decay * theta)
                parameters -= lr * upd
# endregion


# region Adam Optimizer
class Adam(Optimizer):
    """Implements the Adam optimizer."""

    def step(self) -> None:
        """Performs a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            # Get Adam specific parameters from group, else default
            beta1 = group.get("beta1", 0.9)
            beta2 = group.get("beta2", 0.999)
            epsilon = group.get("epsilon", 1e-8)

            for layer, name, parameters in group["params"]:
                grad = getattr(layer, "grad_" + name)
                # Apply weight decay if not zero
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
                v[:] = beta2 * v + (1 - beta2) * (grad ** 2)

                # Compute bias-corrected moment estimates
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # Parameter update
                upd = m_hat / (np.sqrt(v_hat) + epsilon)
                parameters -= lr * upd
# endregion


# region Momentum Optimizer
class Momentum(Optimizer):
    """Implements the Momentum optimizer."""

    def step(self) -> None:
        """Performs a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            # Get parameters specific to Momentum or default
            momentum = group.get("momentum", 0.9)

            for layer, name, parameters in group["params"]:
                grad = getattr(layer, "grad_" + name)
                # Apply weight decay if not zero
                if weight_decay != 0:
                    grad = grad + weight_decay * parameters

                # Initialize per-parameter state if not there
                if (layer, name) not in self.state:
                    self.state[(layer, name)] = {
                        "v": np.zeros_like(parameters)
                    }

                state = self.state[(layer, name)]
                v = state["v"]

                # Update velocity
                v[:] = momentum * v + grad

                # Parameter update
                parameters -= lr * v
# endregion
