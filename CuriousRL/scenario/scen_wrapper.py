from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
from CuriousRL.data import Data
from CuriousRL.utils.Logger import logger
if TYPE_CHECKING:
    from CuriousRL.scenario.dynamic_model.dynamic_model import DynamicModelWrapper

class ScenarioWrapper(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.print_params()

    @abstractmethod
    def reset(self)  -> Data:
        pass

    @abstractmethod
    def step(self, action: List) -> Data:
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def play(self):
        pass

    def print_params(self):
        """Save the parameters of the current scenario in logger.
        """
        logger.info("[+] Scenario Name:" + str(self.name))
        for key in self.kwargs:
            try:
                logger.info("[+] " + key + " = " + str(self.kwargs[key].tolist()))
            except:
                logger.info("[+] " + key + " = " + str(self.kwargs[key]))

    @property
    def name(self):
        """Name of the current scenario."""
        return self.__class__.__name__