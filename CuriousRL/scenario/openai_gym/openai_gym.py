from __future__ import annotations

from torch._C import dtype
from CuriousRL.scenario import Scenario
from torch import Tensor, tensor
import torch
from CuriousRL.utils.Logger import logger
from CuriousRL.data import ActionSpace, Data
import gym
from typing import TYPE_CHECKING, List, Tuple

class OpenAIGym(Scenario):
    def __init__(self, on_gpu:bool, env: gym.Env):
        self._env = env
        self._on_gpu = on_gpu
        if isinstance(self._env.action_space, gym.spaces.Discrete):
            self._action_type = 'Discrete'
            try:
                action_info = self._env.unwrapped.get_action_meanings()
                temp_action_info = "["
                for info in action_info:
                    temp_action_info = temp_action_info + info + ","
                temp_action_info += "]"
            except AttributeError:
                temp_action_info = 'No_action_info'
            self._action_space = ActionSpace(action_range=[list(range(
                self._env.action_space.n))], action_type=['Discrete'], action_info=[temp_action_info])
        elif isinstance(self._env.action_space, gym.spaces.Box):
            self._action_type = 'Box'
            n = self._env.action_space.shape[0]
            try:
                action_info = self._env.unwrapped.get_action_meanings()
            except AttributeError:
                action_info = ['No_action_info']*n
            action_range = []
            for i in range(n):
                action_range.append(
                    [self._env.action_space.low[i], self._env.action_space.high[i]])
            self._action_space = ActionSpace(action_range=action_range, action_type=[
                'Continuous']*n, action_info=action_info)
        else:
            raise Exception('Not support gym space type' +
                            str(self._env.action_space))
        logger.info(env=env)
        obs = env.reset()
        self._state_shape = obs.shape 

    @property
    def on_gpu(self) -> bool:
        return self._on_gpu
    
    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def state_shape(self) -> Tuple:
        return self._state_shape

    def render(self) -> None:
        self._env.render()

    def play(self) -> None:
        raise Exception("Cannot play whole scenario for OpenAI Gym.")

    @property
    def name(self):
        """Name of the current scenario."""
        return "OpenAIGym<" + str(self._env) + ">"

    @property
    def elem(self) -> Data:
        return self.__data

    @property
    def mode(self) -> str: 
        return "single"

    def reset(self) -> OpenAIGym:
        next_state = tensor(self._env.reset())
        if next_state.ndim == 0:
            next_state = next_state.flatten()
        self.__data = Data(next_state=next_state, on_gpu=self._on_gpu)
        return self

    def step(self, action: List) -> OpenAIGym:
        if self._action_type == 'Discrete':
            next_state, reward, done, _ = self._env.step(action[0])
        elif self._action_type == 'Box':
            action = tensor(action, dtype=torch.int).flatten()
            next_state, reward, done, _ = self._env.step(action)
        else:
            next_state, reward, done = None, None, None
            raise Exception("No action type \"" + self._action_type + "\"!")
        reward=torch.tensor(reward)
        next_state = tensor(next_state)
        done = tensor(done, dtype=bool)
        if next_state.ndim == 0:
            next_state = next_state.flatten()
        try:
            self.__data = Data(state=self.elem.next_state,
                                action=action,
                                next_state=next_state,
                                reward=reward,
                                done_flag=done,
                                on_gpu=self._on_gpu)
        except AttributeError:
            logger.error("Must call Scenario.reset() before step!")
            raise Exception("Must call Scenario.reset() before step!")
        return self

