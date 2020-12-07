from __future__ import annotations
from .scen_wrapper import ScenarioWrapper
from CuriousRL.data import ActionSpace, Data
import gym
import numpy as np
from typing import TYPE_CHECKING, List, Tuple

class OpenAIGym(ScenarioWrapper):
    def __init__(self, Env_name):
        super().__init__(Env_name=Env_name)
        self._env = gym.make(Env_name)
        self._current_state = np.asarray(self._env.reset(), dtype=np.float)
        if self._current_state.ndim > 1:
            self._is_state_image = True
        else:
            self._current_state = self._current_state.flatten()
            self._is_state_image = False
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
            self._action_space = ActionSpace(action_range=[list(range(self._env.action_space.n))], action_type=['Discrete'], action_info=[temp_action_info])
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
    
    def reset(self) -> np.array: 
        if self._is_state_image:
            self._current_state = np.asarray(self._env.reset(), dtype=np.float)
        else:
            self._current_state = np.asarray(self._env.reset(), dtype=np.float).flatten()
        return self._current_state 

    def step(self, action: List) -> Data:
        last_state = self._current_state
        if self._action_type == 'Discrete': # not numpy.array, will be int
            action = np.asarray(action).item()
            current_state, reward, done, _ = self._env.step(action)
        elif self._action_type == 'Box':
            action = np.asarray(action).flatten()
            current_state, reward, done, _ = self._env.step(action)
        if self._is_state_image:
            self._current_state = np.asarray(current_state, dtype=np.float)
        else:
            self._current_state = np.asarray(current_state, dtype=np.float).flatten()
        data = Data(state=last_state, action=action, next_state=self._current_state, reward=reward, done_flag=done)
        if done:
            self.reset()
        return data

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
        return self.kwargs['Env_name']

    @property
    def state_shape(self) -> np.array:
        if len(self._current_state.shape) == 0:
            return (1,)
        else:
            return self._current_state.shape

    @property
    def current_state(self) -> np.array:
        return self._current_state