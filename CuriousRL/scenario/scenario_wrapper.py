from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np
from CuriousRL.data import Data, ActionSpace, Batch
from CuriousRL.utils.Logger import logger
from .scenario import Scenario
from typing import TYPE_CHECKING, List, Tuple

class ScenarioWrapper(Scenario):
    def __init__(self, scenario: Scenario):
        self.__scenatio = scenario

    @property
    def action_space(self) -> ActionSpace:
        return self.__scenatio.action_space

    def render(self) -> None:
        self.__scenatio.render()

    def play(self) -> None:
        self.__scenatio.play()

    @property
    def name(self) -> str:
        """Name of the current scenario."""
        last_name = self.__scenatio.name
        return  self.__class__.__name__ + "<" + last_name + ">"

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

    @property
    @abstractmethod
    def state_shape(self) -> Tuple:
        pass