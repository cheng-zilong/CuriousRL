from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np
from CuriousRL.data import Data, ActionSpace
from CuriousRL.utils.Logger import logger
from typing import TYPE_CHECKING, List, Tuple

class ScenarioWrapper(ABC):
    @property
    @abstractmethod
    def state(self) -> Tensor:
        pass

    @property
    @abstractmethod
    def reward(self) -> float:
        pass

    @abstractmethod
    def reset(self)  -> Tensor:
        pass

    @abstractmethod
    def step(self, action: List) -> Data:
        pass

class Scenario(ScenarioWrapper):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.print_params()

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def play(self) -> None:
        pass

    @property
    def name(self):
        """Name of the current scenario."""
        return self.__class__.__name__

    def print_params(self):
        """Save the parameters of the current scenario in logger.
        """
        logger.info("[+] Scenario Name:" + str(self.name))
        for key in self.kwargs:
            try:
                logger.info("[+] " + key + " = " + str(self.kwargs[key].tolist()))
            except:
                logger.info("[+] " + key + " = " + str(self.kwargs[key]))