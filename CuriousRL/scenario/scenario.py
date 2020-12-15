from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Union, TypeVar

from CuriousRL.data import Data, ActionSpace, Batch
from CuriousRL.utils.Logger import logger
TScenario = TypeVar("TScenario", bound="Scenario")

class Scenario(ABC):

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    @abstractmethod
    def mode(self) -> str: 
        pass

    @property
    @abstractmethod
    def action_space(self) -> Union[ActionSpace, List[ActionSpace]]:
        pass

    @property
    @abstractmethod
    def state_shape(self) -> Tuple:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def elem(self) -> Union[Data, Batch]:
        pass

    @property
    @abstractmethod
    def on_gpu(self) -> bool:
        pass

    @abstractmethod
    def reset(self)  -> TScenario:
        pass

    @abstractmethod
    def step(self, action: List) -> TScenario:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def play(self) -> None:
        pass