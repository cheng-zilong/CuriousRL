from __future__ import annotations
import torch
from collections import deque
from gym import spaces
import numpy as np
import gym
import cv2
cv2.ocl.setUseOpenCL(False)
from typing import Tuple

from CuriousRL.data import Data
from CuriousRL.utils.Logger import logger
from CuriousRL.scenario import ScenarioWrapper, Scenario

class AtariScenarioWrapper(ScenarioWrapper):
    def __init__(self, scenario: Scenario, stack_num=4, is_clip_reward=True):
        super().__init__(scenario=scenario)
        self._scenario = scenario
        self._stack_num = stack_num
        self._is_clip_reward = is_clip_reward
        self._frames = deque([], maxlen=stack_num)
        logger.info(name = self.name,
                    stack_num=stack_num,
                    is_clip_reward=is_clip_reward)
        
    def reset(self) -> Scenario:
        self._scenario.reset()
        state = torch.transpose(self._scenario.elem.next_state, 0, 2)
        for _ in range(self._stack_num):
            self._frames.append(state)
        new_state = torch.cat(tuple(self._frames), dim=0)
        self.__data = Data(next_state=new_state, on_gpu=self.on_gpu)
        return self

    def step(self, action) -> Scenario:
        self._scenario.step(action)
        new_reward = torch.sign(
            self._scenario.elem.reward) if self._is_clip_reward else self._scenario.elem.reward  # clip reward
        self._frames.append(torch.transpose(
            self._scenario.elem.next_state, 0, 2))
        new_state = torch.cat(tuple(self._frames), dim=0)
        self.__data = Data(state=self.__data.next_state,
                           action=self._scenario.elem.action,
                           next_state=new_state,
                           reward=new_reward,
                           done_flag=self._scenario.elem.done_flag, 
                           on_gpu=self.on_gpu)
        return self

    @property
    def elem(self) -> Data:
        return self.__data

    @property
    def state_shape(self) -> Tuple:
        old_shape = self._scenario.state_shape
        new_shape = (self._stack_num, old_shape[1], old_shape[0])
        return new_shape

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env:gym.Env, noop_max:int=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env:gym.Env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env:gym.Env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env:gym.Env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._obs_buffer.max(axis=0)
        else:
            return self.env.render(mode)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env:gym.Env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

def wrap_deepmind(env_name, episode_life=True):
    """Configure environment for DeepMind-style Atari."""
    env = gym.make(env_name)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    return env
