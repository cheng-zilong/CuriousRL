from __future__ import annotations
from os import name
from loguru import logger
import torch
from typing import List, Tuple, Union
from torch import tensor, Tensor
from torch._C import dtype
import torch.multiprocessing as mp

from .scenario_wrapper import ScenarioWrapper
from .scenario import Scenario
from CuriousRL.utils.config import global_config
from CuriousRL.data import Data, ActionSpace, Batch

def _worker(index, scenario: Scenario, share_memmory_batch: Batch, curr_state:Tensor, pipe, seed):
    global_config.set_random_seed(seed + index)
    while True:
        command, action = pipe.recv()
        if command == 'reset':
            curr_state[index] = scenario.reset()
            pipe.send(True)  # Send Done
        elif command == 'step':
            share_memmory_batch[index] = scenario.step(action)
            curr_state[index] = scenario.curr_state
            pipe.send(True)  # Send Done
        elif command == 'render':
            scenario.render()
        else:
            raise RuntimeError(
                'Received unknown command `{0}`.'.format(command))

class ScenaroAsync(ScenarioWrapper):
    def __init__(self,
                 scenario: Scenario,
                 num: int,
                 context: str = 'spawn'):
        super().__init__(scenario=scenario)
        self._action_space = [scenario.action_space] * num
        self._state_shape = (num, *scenario.curr_state.shape)
        dummy_state = scenario.reset()
        self._curr_state = torch.zeros(self._state_shape, dtype =dummy_state.dtype).share_memory_()
        self._share_memmory_batch = Batch(on_gpu=False,
                                    state=torch.zeros(
                                        self._state_shape, dtype=dummy_state.dtype),
                                    action=torch.zeros(
                                        (num, len(scenario.action_space))),
                                    next_state=torch.zeros(
                                        self._state_shape, dtype=dummy_state.dtype),
                                    reward=torch.zeros(num),
                                    done_flag=torch.zeros(num, dtype=torch.bool)).share_memmory_()
        ctx = mp.get_context(context)
        self._name = "ScenaroAsync<" + scenario.name + ">"
        self._parent_pipes, self._processes = [], []
        logger.info("[+ +] start workers...")
        for index in range(num):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(target=_worker,
                                  name='Worker<{0}>-({1})'.format(
                                      scenario.name, index),
                                  args=(index,
                                        scenario,
                                        self._share_memmory_batch,
                                        self._curr_state,
                                        child_pipe,
                                        global_config.random_seed))
            self._parent_pipes.append(parent_pipe)
            self._processes.append(process)
            process.start()
            logger.info("[+ +] Worker<%s>-%d starts!" % (scenario.name, index))
        self._curr_state = self.reset()

    @property
    def action_space(self) -> List[ActionSpace]:
        return self._action_space

    def render(self) -> None:
        for pipe in self._parent_pipes:
            pipe.send(('render', None))

    def play(self) -> None:
        raise Exception('ScenaroAsync instance cannot play.')

    @property
    def name(self) -> str:
        """Name of the current scenario."""
        return self._name

    @property
    def curr_state(self) -> Tensor:
        return self._curr_state

    def reset(self) -> Tensor:
        for pipe in self._parent_pipes:
            pipe.send(('reset', None))
        success_flags = [pipe.recv() for pipe in self._parent_pipes]
        if all(success_flags):
            return self._curr_state

    def step(self, action: List) -> Batch:
        for (index, pipe) in enumerate(self._parent_pipes):
            pipe.send(('step', action[index]))
        success_flags = [pipe.recv() for pipe in self._parent_pipes]
        if all(success_flags):
            return self._share_memmory_batch

    @property
    def mode(self) -> str:
        return "multiple"