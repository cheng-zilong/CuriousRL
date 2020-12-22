from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np
from CuriousRL.data import Data, ActionSpace, Batch
from CuriousRL.utils.Logger import logger
from .scenario import Scenario
from typing import TYPE_CHECKING, List, Tuple, Union

class ScenarioWrapper(Scenario):
    def __init__(self, scenario: Scenario):
        self.__scenatio = scenario

    @property
    def on_gpu(self) -> bool:
        return self.__scenatio.on_gpu

    @property
    def action_space(self)  -> Union[ActionSpace, List[ActionSpace]]:
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
    def curr_state(self) -> Tensor:
        return self.__scenatio.curr_state

    @abstractmethod
    def reset(self)  -> Tensor:
        return self.__scenatio.reset()

    @abstractmethod
    def step(self, action: List)  -> Union[Data, Batch]:
        return self.__scenatio.reset(action)