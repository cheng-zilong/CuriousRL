from __future__ import annotations
from os import name
from loguru import logger
import torch
from typing import List, Tuple
from torch._C import dtype
import torch.multiprocessing as mp
import platform

from .scenario_wrapper import ScenarioWrapper
from .scenario import Scenario
from CuriousRL.utils.config import global_config
from CuriousRL.data import Data, ActionSpace, Batch

def _worker(index, scenario: Scenario, share_memmory_batch: Batch, pipe, seed):
    global_config.set_random_seed(seed + index)
    while True:
        command, action = pipe.recv()
        if command == 'reset':
            scenario.reset()
            pipe.send(True)  # Send Done
        elif command == 'step':
            elem = scenario.step(action).elem
            share_memmory_batch[index] = elem
            if elem.done_flag:
                scenario.reset()
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
        self._state_shape = (num, *scenario.state_shape)
        dummy_elem = scenario.reset().elem
        if platform.system() == "Windows":
            self._share_memmory_batch = Batch(on_gpu=False,
                                        state=torch.zeros(
                                            self._state_shape, dtype=dummy_elem.next_state.dtype),
                                        action=torch.zeros(
                                            (num, len(scenario.action_space))),
                                        next_state=torch.zeros(
                                            self._state_shape, dtype=dummy_elem.next_state.dtype),
                                        reward=torch.zeros(num),
                                        done_flag=torch.zeros(num, dtype=torch.bool)).share_memmory_()
        else:
            self._share_memmory_batch = Batch(on_gpu=scenario.on_gpu,
                                        state=torch.zeros(
                                            self._state_shape, dtype=dummy_elem.next_state.dtype),
                                        action=torch.zeros(
                                            (num, len(scenario.action_space))),
                                        next_state=torch.zeros(
                                            self._state_shape, dtype=dummy_elem.next_state.dtype),
                                        reward=torch.zeros(num),
                                        done_flag=torch.zeros(num, dtype=torch.bool)).share_memmory_()

        ctx = mp.get_context(context)
        self._name = "ScenaroAsync<" + scenario.name + ">"
        self._parent_pipes, self._processes = [], []
        logger.info("[+ +] start workers...")
        for index in range(num):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(target=_worker,
                                  name='Worker<{0}>-{1}'.format(
                                      scenario.name, index),
                                  args=(index,
                                        scenario,
                                        self._share_memmory_batch,
                                        child_pipe,
                                        global_config.random_seed))
            self._parent_pipes.append(parent_pipe)
            self._processes.append(process)
            process.start()
            logger.info("[+ +] Worker<%s>-%d starts!" % (scenario.name, index))

    @property
    def action_space(self) -> List[ActionSpace]:
        return self._action_space

    @property
    def state_shape(self) -> Tuple:
        return self._state_shape

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
    def elem(self) -> Batch:
        return self._share_memmory_batch

    def reset(self) -> ScenaroAsync:
        for pipe in self._parent_pipes:
            pipe.send(('reset', None))
        success_flags = [pipe.recv() for pipe in self._parent_pipes]
        if all(success_flags):
            return self

    def step(self, action: List) -> ScenaroAsync:
        for (index, pipe) in enumerate(self._parent_pipes):
            pipe.send(('step', action[index]))
        success_flags = [pipe.recv() for pipe in self._parent_pipes]
        if all(success_flags):
            return self

    @property
    def mode(self) -> str:
        return "multiple"