from __future__ import annotations

from torch import Tensor, tensor
import torch
import gym
from typing import TYPE_CHECKING, List, Tuple

from CuriousRL.utils.Logger import logger
from CuriousRL.data import ActionSpace, Data
from CuriousRL.scenario import Scenario

class OpenAIGym(Scenario):
    def __init__(self, env: gym.Env, on_gpu: bool=False):
        self.__env = env
        self.__on_gpu = on_gpu
        if isinstance(self.__env.action_space, gym.spaces.Discrete):
            self._action_type = 'Discrete'
            try:
                action_info = self.__env.unwrapped.get_action_meanings()
                temp_action_info = "["
                for info in action_info:
                    temp_action_info = temp_action_info + info + ","
                temp_action_info += "]"
            except AttributeError:
                temp_action_info = 'No_action_info'
            self.__action_space = ActionSpace(action_range=[list(range(
                self.__env.action_space.n))], action_type=['Discrete'], action_info=[temp_action_info])
        elif isinstance(self.__env.action_space, gym.spaces.Box):
            self._action_type = 'Box'
            n = self.__env.action_space.shape[0]
            try:
                action_info = self.__env.unwrapped.get_action_meanings()
            except AttributeError:
                action_info = ['No_action_info']*n
            action_range = []
            for i in range(n):
                action_range.append(
                    [self.__env.action_space.low[i], self.__env.action_space.high[i]])
            self.__action_space = ActionSpace(action_range=action_range, action_type=[
                'Continuous']*n, action_info=action_info)
        else:
            raise Exception('Not support gym space type' +
                            str(self.__env.action_space))
        logger.info(env=env)
        self.__curr_state = tensor(self.__env.reset())

    @property
    def on_gpu(self) -> bool:
        return self.__on_gpu
    
    @property
    def action_space(self) -> ActionSpace:
        return self.__action_space

    def render(self) -> None:
        self.__env.render()

    def play(self) -> None:
        raise Exception("Cannot play whole scenario for OpenAI Gym.")

    @property
    def name(self):
        """Name of the current scenario."""
        return "OpenAIGym<" + str(self.__env) + ">"

    @property
    def curr_state(self) -> Tensor:
        return self.__curr_state

    @property
    def mode(self) -> str: 
        return "single"

    def reset(self) -> OpenAIGym:
        self.__curr_state = tensor(self.__env.reset())
        if self.__curr_state.ndim == 0:
            self.__curr_state = self.__curr_state.flatten()
        return self.__curr_state

    def step(self, action: List) -> Data:
        if self._action_type == 'Discrete':
            next_state, reward, done, _ = self.__env.step(action[0])
        elif self._action_type == 'Box':
            action = tensor(action, dtype=torch.int).flatten()
            next_state, reward, done, _ = self.__env.step(action)
        else:
            next_state, reward, done = None, None, None
            raise Exception("No action type \"" + self._action_type + "\"!")
        reward=torch.tensor(reward)
        done = tensor(done, dtype=bool)
        if next_state.ndim == 0:
            next_state = next_state.flatten()
        data = Data(    state=self.__curr_state,
                        action=action,
                        next_state=next_state,
                        reward=reward,
                        done_flag=done,
                        on_gpu=self.__on_gpu)
        if done == True:
            self.__curr_state = self.__env.reset()
        else:
            self.__curr_state = next_state
        return data

