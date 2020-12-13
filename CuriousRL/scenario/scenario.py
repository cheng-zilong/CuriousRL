from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union

from CuriousRL.data import Data, ActionSpace, Batch
from CuriousRL.utils.Logger import logger

class Scenario(ABC):
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        pass

    @property
    @abstractmethod
    def state_shape(self) -> Tuple:
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

    @property
    @abstractmethod
    def elem(self) -> Union[Data, Batch]:
        pass

    @abstractmethod
    def reset(self)  -> Scenario:
        pass

    @abstractmethod
    def step(self, action: List) -> Scenario:
        pass