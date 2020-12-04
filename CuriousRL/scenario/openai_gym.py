from .scen_wrapper import ScenarioWrapper
from CuriousRL.data import ActionSpace, Data
import gym
import numpy as np
from typing import TYPE_CHECKING, List


class OpenAIGym(ScenarioWrapper):
    def __init__(self, Env_name):
        super().__init__(Env_name=Env_name)
        self._env = gym.make(Env_name)
        self._current_state = self._env.reset()
        action_example = self._env.action_space.sample()
        if isinstance(action_example, np.ndarray):
            self._action_shape = action_example.shape
            self._action_type = action_example.dtype
        else:
            self._action_type = None # not numpy.ndarray, will be int

    def reset(self) -> Data:
        self._current_state = self._env.reset()

    def step(self, action: List) -> Data:
        if self._action_type is None: # not numpy.ndarray, will be int
            action = np.asarray(action).item()
        else:
            action = np.asarray(action, dtype=self._action_type).reshape(self._action_shape)
        last_state = self._current_state
        self._current_state, reward, done, _ = self._env.step(action)
        data = Data(state=last_state, action=action, next_state=self._current_state, reward=reward, done_flag=done)
        if done:
            self.reset()
        return data

    @property
    def action_space(self) :
        gym_action_space = self._env.action_space
        if isinstance(gym_action_space, gym.spaces.Discrete):
            n = gym_action_space.n
            try:
                action_info = self._env.unwrapped.get_action_meanings()
            except AttributeError:
                action_info = ['No_action_info']*n
            temp_action_space = ActionSpace(action_range=[list(range(n))], action_type=[
                                            'Discrete']*n, action_info=action_info)
        elif isinstance(gym_action_space, gym.spaces.Box):
            n = gym_action_space.shape[0]
            try:
                action_info = self._env.unwrapped.get_action_meanings()
            except AttributeError:
                action_info = ['No_action_info']*n
            action_range = []
            for i in range(n):
                action_range.append(
                    [self._env.action_space.low[i], self._env.action_space.high[i]])
            temp_action_space = ActionSpace(action_range=action_range, action_type=[
                                            'Continuous']*n, action_info=action_info)
        else:
            raise Exception('Not support gym space type' +
                            str(type(gym_action_space)))
        return temp_action_space

    def render(self):
        self._env.render()

    def play(self):
        pass

    @property
    def name(self):
        """Name of the current scenario."""
        return self.kwargs['Env_name']
