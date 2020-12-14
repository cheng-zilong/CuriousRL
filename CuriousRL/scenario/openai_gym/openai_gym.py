from __future__ import annotations
from CuriousRL.scenario import Scenario
from torch import Tensor, tensor
import torch
from CuriousRL.utils.config import global_config
from CuriousRL.utils.Logger import logger
from CuriousRL.data import ActionSpace, Data
import gym
import numpy as np
from typing import TYPE_CHECKING, List, Tuple

class OpenAIGym(Scenario):
    def __init__(self, env: gym.Env):
        self._env = env
        obs = env.reset()
        self._state_shape = obs.shape 
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
                            str(type(gym_action_space)))
        logger.info(env=env)
        
    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    @property
    def state_shape(self) -> Tuple:
        return self._state_shape

    def render(self):
        self._env.render()

    def play(self):
        raise Exception("Cannot play whole scenario for OpenAI Gym.")

    @property
    def name(self):
        """Name of the current scenario."""
        return "OpenAIGym<" + str(self._env) + ">"

    @property
    def elem(self) -> Data:
        return self.__data

    def reset(self) -> Scenario:
        next_state = tensor(self._env.reset(), dtype=torch.float)
        if next_state.ndim == 0:
            next_state = next_state.view(-1)
        self.__data = Data(next_state=next_state)
        return self

    def step(self, action: List) -> Scenario:
        if self._action_type == 'Discrete':
            next_state, reward, done, _ = self._env.step(action[0])
        elif self._action_type == 'Box':
            action = tensor(action, dtype=torch.float).view(-1)
            next_state, reward, done, _ = self._env.step(action)
        next_state = tensor(next_state, dtype=torch.float)
        if next_state.ndim == 0:
            next_state = next_state.view(-1)
        self.__data = Data(state=self.elem.next_state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                    done_flag=done)
        return self
