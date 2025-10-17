# SGD, Momentum, Adam

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from .layers import Layer


class Optimizer(ABC):
    def __init__(
        self,
        params: list[Tuple[Layer, str, np.ndarray]] | list[Dict],
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ) -> None:
        """
        `params` can be:
        - an iterable of triples (layer, name, array) where layer is a Layer instance, name is the parameter name (str), and array is the parameter tensor (np.ndarray)
        - OR a list of 'param groups': dicts with params and per-group hyperparameters
        """
        self.param_groups = self._make_param_groups(params, lr, weight_decay)
        self.state = {}  # per-parameter state dictionaries

    @abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        """
        Sets all parameter gradients to zero.
        """
        for group in self.param_groups:
            for layer, name, param in group["params"]:
                grad = getattr(layer, "grad_" + name)
                grad.fill(0.0)

    def _make_param_groups(
        self,
        params: list[Tuple[Layer, str, np.ndarray]] | list[Dict],
        lr: float,
        weight_decay: float,
    ) -> list[Dict[str, Any]]:
        pass
