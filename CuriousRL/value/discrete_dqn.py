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
                 eps_linear_decay_len=1e6,
                 eps_exp_decay_rate=0.999,
                 gamma_discount=0.99,
                 target_replace_steps=1000,
                 buffer_size=1e4,
                 max_episode=1e3,
                 network=None,
                 optimizer=None,
                 is_render=False):
        self._steps_counter = 0
        self._network = network
        self._optimizer = optimizer
        self._eps_linear_decay_step = (eps_start-eps_end)/eps_linear_decay_len
        self._eps = eps_start
        super().__init__(batch_size=int(batch_size),
                         eps_start=eps_start,
                         eps_end=eps_end,
                         eps_linear_decay_len=int(eps_linear_decay_len),
                         eps_exp_decay_rate=eps_exp_decay_rate,
                         gamma_discount=gamma_discount,
                         target_replace_steps=int(target_replace_steps),
                         buffer_size=int(buffer_size),
                         max_episode=int(max_episode),
                         network=network,
                         optimizer=optimizer,
                         is_render=is_render)

    def init(self, scenario: Scenario) -> DiscreteDQN:
        self._scenario = scenario
        reset_state = scenario.reset()
        if self._network == None:
            if scenario.state.ndim > 1:  # is image
                self._network = example_network.ThreeLayerConvolutionalNetwork(reset_state.shape[0], len(scenario.action_space._action_range[0]))
            else:  # not image
                self._network = example_network.TwoLayerAllConnetedNetwork(reset_state.shape[0], len(scenario.action_space._action_range[0]))
        self._dataset = Dataset(buffer_size=self.kwargs['buffer_size'])
        self._eval_net = self._network
        if self._optimizer == None:
            self._optimizer = torch.optim.Adam(
                params=self._eval_net.parameters(), lr=0.001)
        self._target_net = copy.deepcopy(self._eval_net)
        if global_config.is_cuda:
            self._eval_net = self._eval_net.cuda()
            self._target_net = self._target_net.cuda()
        self._loss_func = nn.MSELoss()
        self._eval_net.eval()
        self._target_net.eval()
        return self

    def _choose_action(self, x):
        if self._steps_counter < self.kwargs['eps_linear_decay_len']:
            self._eps -= self._eps_linear_decay_step
        elif self._steps_counter == self.kwargs['eps_linear_decay_len']:
            logger.info(
                "[+] ------------- EXPONENTIAL MODE START ---------------------")
        else:
            self._eps *= self.kwargs['eps_exp_decay_rate']
        if np.random.random() > self._eps:
            action = self._eval_net(x.unsqueeze(0)).max(1)[1].item()
        else:
            action = self._scenario.action_space.sample()
        return action

    def _learn(self, target_replace_steps, batch_size, gamma):
        self._eval_net.train()
        self._target_net.train()
        if self._steps_counter % target_replace_steps == 0:
            self._target_net.load_state_dict(self._eval_net.state_dict())
        sample_data = self._dataset.fetch_random_data(batch_size)
        q_eval = self._eval_net(sample_data.state).gather(
            1, sample_data.action.long().unsqueeze(1))
        with torch.no_grad():
            q_next = self._target_net(sample_data.next_state)
            q_target = sample_data.reward + \
                (~sample_data.done_flag) * gamma * q_next.max(1)[0]
        loss = self._loss_func(q_eval, q_target.unsqueeze(1))
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._eval_net.eval()
        self._target_net.eval()

    def solve(self):
        batch_size = self.kwargs['batch_size']
        gamma = self.kwargs['gamma_discount']
        is_render = self.kwargs['is_render']
        target_replace_steps = self.kwargs['target_replace_steps']
        for epi in range(self.kwargs['max_episode']):
            episode_reward = 0.
            while True:
                action = self._choose_action(self._scenario.state)
                new_data = self._scenario.step(action)  # take action
                episode_reward += new_data.reward
                if is_render:
                    self._scenario.render()
                self._dataset.update(new_data)
                if self._dataset.current_buffer_size > batch_size:
                    self._learn(target_replace_steps, batch_size, gamma)
                self._steps_counter += 1
                if new_data.done_flag == True:
                    break
            logger.info('[+ +] Total Steps:%8d\t ' % (self._steps_counter) + 'Episode:%5d' % (epi) + '\t Episode Reward:%5d' %
                        (episode_reward) + '\t Epsilon:%.5f' % (self._eps))
