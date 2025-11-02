import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from .layers.base_layers import Layer


# region Abstract Optimizer
class Optimizer(ABC):
    """Base class for optimizers using a dict of parameters."""

    def __init__(
        self,
        raw_params: Dict[str, np.ndarray],
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize the optimizer.

        Args:
            raw_params: Dict of parameter_name -> np.ndarray
            lr: Learning rate
            weight_decay: L2 regularization factor
        """
        self.param_groups = self._make_param_groups(raw_params, lr, weight_decay)
        self.state: Dict[Tuple[None, str], Dict[str, np.ndarray]] = {}

    @abstractmethod
    def step(self) -> None:
        """Perform a single optimization step."""
        pass

    def zero_grad(self) -> None:
        """Set all gradients to zero."""
        for group in self.param_groups:
            for _, name, param in group["params"]:
                if hasattr(param, "grad") and param.grad is not None:
                    param.grad.fill(0.0)

    def _make_param_groups(
        self,
        raw_params: Dict[str, np.ndarray],
        lr: float,
        weight_decay: float,
    ) -> List[Dict[str, Any]]:
        """Convert a dict of parameters into a standard param group."""
        flat_params: List[Tuple[None, str, np.ndarray]] = [
            (None, name, param) for name, param in raw_params.items()
        ]
        return [{"params": flat_params, "lr": lr, "weight_decay": weight_decay}]
# endregion


# region SGD Optimizer
class SGD(Optimizer):
    """Stochastic Gradient Descent."""

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for _, name, param in group["params"]:
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                param -= lr * grad


# endregion


# region Adam Optimizer
class Adam(Optimizer):
    """Adam optimizer."""

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group.get("beta1", 0.9)
            beta2 = group.get("beta2", 0.999)
            epsilon = group.get("epsilon", 1e-8)

            for _, name, param in group["params"]:
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                if (None, name) not in self.state:
                    self.state[(None, name)] = {
                        "m": np.zeros_like(param),
                        "v": np.zeros_like(param),
                        "t": 0,
                    }

                state = self.state[(None, name)]
                m, v, t = state["m"], state["v"], state["t"]

                t += 1
                state["t"] = t

                m[:] = beta1 * m + (1 - beta1) * grad
                v[:] = beta2 * v + (1 - beta2) * (grad**2)

                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                param -= lr * (m_hat / (np.sqrt(v_hat) + epsilon))


# endregion


# region Momentum Optimizer
class Momentum(Optimizer):
    """SGD with momentum."""

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group.get("momentum", 0.9)

            for _, name, param in group["params"]:
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue

                if weight_decay != 0:
                    grad = grad + weight_decay * param

                if (None, name) not in self.state:
                    self.state[(None, name)] = {"v": np.zeros_like(param)}

                v = self.state[(None, name)]["v"]
                v[:] = momentum * v + grad
                param -= lr * v


# endregion
