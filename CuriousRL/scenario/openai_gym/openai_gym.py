from __future__ import annotations
from CuriousRL.scenario import Scenario
from torch import Tensor, tensor
import torch
from CuriousRL.utils.config import global_config
from CuriousRL.data import ActionSpace, Data
import gym
import numpy as np
from typing import TYPE_CHECKING, List, Tuple

class OpenAIGym(Scenario):
    def __init__(self, env: gym.Env):
        self._env = env
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
        super().__init__(env=env)

    @property
    def action_space(self) -> ActionSpace:
        return self._action_space

    def render(self):
        self._env.render()

    def play(self):
        raise Exception("Cannot play whole scenario for OpenAI Gym.")

    @property
    def name(self):
        """Name of the current scenario."""
        return self._env.unwrapped.spec.id

    ########################################
    ## Following methods can be override ###
    ########################################
    @property
    def state(self) -> Tensor:
        return self.__state

    @property
    def reward(self) -> float:
        return self.__reward

    def reset(self) -> Tensor:
        self.__state = tensor(self._env.reset(), dtype=torch.float)
        if self.__state.ndim == 0:
            self.__state = self.__state.flatten()
        if global_config.is_cuda:
            self.__state = self.__state.cuda()
        return self.__state

    def step(self, action: List) -> Data:
        last_state = self.__state
        if self._action_type == 'Discrete':  # not numpy.array, will be int
            action = tensor(action, dtype=torch.int).item()
            state, self.__reward, done, _ = self._env.step(action)
        elif self._action_type == 'Box':
            action = tensor(action, dtype=torch.float).flatten()
            state, self.__reward, done, _ = self._env.step(action)
        self.__state = tensor(state, dtype=torch.float)
        if self.__state.ndim == 0:
            self.__state = self.__state.flatten()
        if global_config.is_cuda:
            self.__state = self.__state.cuda()
        data = Data(state=last_state,
                    action=action,
                    next_state=self.__state,
                    reward=self.__reward,
                    done_flag=done)
        if done:
            self.reset()
        return data