from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import gym
from CuriousRL.scenario import Scenario
from CuriousRL.utils.config import global_config
from CuriousRL.utils.Logger import logger
from CuriousRL.algorithm import Algorithm
from CuriousRL.data import Data, Dataset
from . import example_network
import time as tm
import copy


class DiscreteDQN(Algorithm):
    """ This class is to build DQN algorithm."""

    def __init__(self,
                 batch_size=32,
                 eps_start=1,
                 eps_end=0.05,
                 eps_linear_decay_len=1e4,
                 eps_exp_decay_rate=0.999,
                 gamma=0.999,
                 target_replace_frames=1000,
                 buffer_size=1e4,
                 max_episode=1e3,
                 network=None,
                 optimizer=None,
                 is_render=False):
        self._frames_counter = 0
        self._network = network
        self._optimizer = optimizer
        self._eps_linear_decay_step = (eps_start-eps_end)/eps_linear_decay_len
        self._eps = eps_start
        self._eps_linear_decay_len = int(eps_linear_decay_len)
        self._eps_exp_decay_rate = eps_exp_decay_rate
        self._batch_size = int(batch_size)
        self._gamma = gamma
        self._target_replace_frames = int(target_replace_frames)
        self._buffer_size = buffer_size
        self._max_episode = max_episode
        self._is_render = is_render
        logger.info(batch_size=int(batch_size),
                    eps_start=eps_start,
                    eps_end=eps_end,
                    eps_linear_decay_len=int(eps_linear_decay_len),
                    eps_exp_decay_rate=eps_exp_decay_rate,
                    gamma=gamma,
                    target_replace_frames=int(target_replace_frames),
                    buffer_size=int(buffer_size),
                    max_episode=int(max_episode),
                    network=network,
                    optimizer=optimizer,
                    is_render=is_render)

    def init(self, scenario: Scenario) -> DiscreteDQN:
        self._scenario = scenario
        reset_state = scenario.reset().elem.next_state
        if self._network is None:
            if reset_state.ndim > 1:  # is image
                self._network = example_network.ThreeLayerConvolutionalNetwork(
                    reset_state.shape[0], len(scenario.action_space._action_range[0]))
            else:  # not image
                self._network = example_network.TwoLayerAllConnetedNetwork(
                    reset_state.shape[0], len(scenario.action_space._action_range[0]))
        self._dataset = Dataset(buffer_size=self._buffer_size)
        self._eval_net = self._network
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                params=self._eval_net.parameters(), lr=1e-4)
        self._target_net = copy.deepcopy(self._eval_net)
        if global_config.is_cuda:
            self._eval_net = self._eval_net.cuda()
            self._target_net = self._target_net.cuda()
        self._loss_func = nn.MSELoss()
        return self

    def _choose_action(self, x):
        if np.random.random() > self._eps:
            action = [self._eval_net(x.unsqueeze(0)).max(1)[1].item()]
        else:
            action = self._scenario.action_space.sample()
        return action

    def _learn(self):
        if self._frames_counter % self._target_replace_frames == 0:
            self._target_net.load_state_dict(self._eval_net.state_dict())
        sample_data = self._dataset.fetch_random_data(self._batch_size)
        q_eval = self._eval_net(sample_data.state).gather(
            1, sample_data.action.long())
        with torch.no_grad():
            q_next = self._target_net(sample_data.next_state)
            q_target = sample_data.reward + \
                (~sample_data.done_flag) * self._gamma * q_next.max(1)[0]
        loss = self._loss_func(q_eval, q_target.unsqueeze(1))
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def solve(self):
        total_start_time = tm.time()
        for epi in range(self._max_episode):
            episode_reward = 0.
            epi_start_time = tm.time()
            while True:
                if self._frames_counter < self._eps_linear_decay_len:
                    self._eps -= self._eps_linear_decay_step
                elif self._frames_counter == self._eps_linear_decay_len:
                    logger.info(
                        "[+] ------------- EXPONENTIAL MODE START ---------------------")
                else:
                    self._eps *= self._eps_exp_decay_rate
                action = self._choose_action(self._scenario.elem.next_state)
                new_data = self._scenario.step(action).elem  # take action
                episode_reward += new_data.reward
                if self._is_render:
                    self._scenario.render()
                self._dataset.update(new_data)
                if self._dataset.current_buffer_size > self._batch_size:
                    self._learn()
                self._frames_counter += 1
                if new_data.done_flag:
                    self._scenario.reset()
                    break
            epi_end_time = tm.time()
            logger.info('[+ +]  Total Frames:%8d' % (self._frames_counter) +
                        '\t Episode:%5d' % (epi) +
                        '\t Episode Reward:%5d' % (episode_reward) +
                        '\t Epsilon:%.5f' % (self._eps) +
                        '\t Episode Time:%.5f' % (epi_end_time - epi_start_time) +
                        '\t Total Time:%.5f' % (epi_end_time - total_start_time))
