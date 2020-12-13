from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor
import numpy as np
from typing import TYPE_CHECKING, List, Tuple
import torch.multiprocessing as mp

from .scenario_wrapper import ScenarioWrapper
from .scenario import Scenario
from CuriousRL.utils.config import global_config
from CuriousRL.data import Data, ActionSpace, Batch
from CuriousRL.utils.Logger import logger

def _worker(index, scenario:Scenario, pipe, parent_pipe, seed):
    global_config.set_random_seed(seed + index)
    parent_pipe.close()
    while True:
        command, action = pipe.recv()
        if command == 'reset':
            scenario.reset()
        elif command == 'step':
            elem = scenario.step(action).elem
            pipe.send(elem)
            if elem.done_flag:
                scenario.reset()
        else:
            raise RuntimeError('Received unknown command `{0}`. Must '
                'be one of {`reset`, `step`, `seed`, `close`, '
                '`_check_observation_space`}.'.format(command))
        del command
        del action

class ScenaroAsync(ScenarioWrapper):
    def __init__(   self,
                    scenario: Scenario,
                    num: int,
                    context: str ='spawn'):
        action_range = scenario.action_space.action_range * num
        action_type = scenario.action_space.action_type * num
        action_info = scenario.action_space.action_info * num
        self._action_space = ActionSpace(action_range=action_range, action_type=action_type, action_info=action_info)
        self._state_shape = scenario.state_shape

        ctx = mp.get_context(context)
        self._name = "ScenaroAsync<" + scenario.name + ">"
        self._parent_pipes, self._processes = [], []
        for index in range(num):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(target=_worker,
                name='Worker<{0}>-{1}'.format(scenario.name, index),
                args=(index, scenario, child_pipe, parent_pipe, global_config.random_seed))
            self._parent_pipes.append(parent_pipe)
            self._processes.append(process)
            process.start()
            child_pipe.close()

    @property
    def action_space(self) -> ActionSpace:
        self._action_space

    @property
    def state_shape(self) -> Tuple:
        return self._state_shape

    def render(self) -> None:
        raise Exception("ScenaroAsync instance cannot render.")

    def play(self) -> None:
        raise Exception("ScenaroAsync instance cannot play.")

    @property
    def name(self) -> str:
        """Name of the current scenario."""
        return self._name

    @property
    @abstractmethod
    def elem(self) -> Batch:
        pass

    @abstractmethod
    def reset(self)  -> Scenario:
        for pipe in self._parent_pipes:
            pipe.send(('reset', None))

    @abstractmethod
    def step(self, action: List) -> Scenario:
        pass




