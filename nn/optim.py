import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List


# region Abstract Optimizer
class Optimizer(ABC):
    """Base class for optimisers using a dictionary of parameters.

    This class serves as the base class for all optimizers, providing the
    interface and common functionality required for optimisation algorithms.
    """
    def __init__(
        self,
        raw_params: Dict[str, np.ndarray],
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialise the optimizer with the given parameters.

        Args:
            raw_params (Dict[str, np.ndarray]): A dictionary mapping parameter
                                                names to their corresponding
                                                arrays.
            lr (float, optional): The learning rate. Defaults to 0.001.
            weight_decay (float, optional): The L2 regularization factor.
                                            Defaults to 0.0.
        """
        self.param_groups = self._make_param_groups(raw_params, lr,
                                                    weight_decay)
        self.state: Dict[Tuple[None, str], Dict[str, np.ndarray]] = {}

    @abstractmethod
    def step(self) -> None:
        """Perform a single optimization step.

        This method updates the parameters based on their gradients and the
        specified optimization algorithm.
        """
        pass

    def zero_grad(self) -> None:
        """Set all gradients to zero.

        This method resets the gradients of all parameters to zero, which is
        typically called before computing new gradients for the next batch of
        data.
        """
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
        """Convert a dictionary of parameters into a standard parameter group.

        This method organizes the parameters into groups for optimization,
        applying the specified learning rate and weight decay.

        Args:
            raw_params (Dict[str, np.ndarray]): A dictionary mapping parameter
                                                names to their corresponding
                                                arrays.
            lr (float): The learning rate.
            weight_decay (float): The L2 regularization factor.

        Returns:
            List[Dict[str, Any]]: A list of parameter groups, each containing
                                  parameters, learning rate, and weight decay.
        """
        flat_params: List[Tuple[None, str, np.ndarray]] = [
            (None, name, param) for name, param in raw_params.items()
        ]
        return [{"params": flat_params, "lr": lr,
                 "weight_decay": weight_decay}]

# endregion


# region SGD Optimizer
class SGD(Optimizer):
    """Stochastic Gradient Descent optimiser.

    This class implements the Stochastic Gradient Descent optimisation
    algorithm, which updates the parameters in the direction opposite to the
    gradient of the loss function.
    """

    def step(self) -> None:
        """Perform a single optimisation step.

        This method updates the parameters using the Stochastic Gradient
        Descent algorithm, adjusting each parameter by the negative gradient
        scaled by the learning rate.
        """
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
    """Adam optimiser.

    This class implements the Adam optimisation algorithm, which combines the
    benefits of AdaGrad and RMSProp, using estimates of both the first and
    second moments of the gradients.
    """

    def step(self) -> None:
        """Perform a single optimization step.

        This method updates the parameters using the Adam optimization
        algorithm, which involves computing the moving averages of the
        gradients and their squares, and then adjusting the parameters
        accordingly.
        """
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
    """Stochastic Gradient Descent with momentum.

    This class implements the SGD with momentum optimisation algorithm, which
    accelerates the gradient vectors in the direction of their moving average,
    helping to speed up convergence and reduce oscillations.
    """

    def step(self) -> None:
        """Perform a single optimisation step.

        This method updates the parameters using the Stochastic Gradient
        Descent with momentum algorithm, which involves computing a moving
        average of the gradients and adjusting the parameters accordingly.
        """
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
