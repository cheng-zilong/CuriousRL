from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import gym
from CuriousRL.scenario import ScenarioWrapper
from CuriousRL.utils.config import global_config
from CuriousRL.utils.Logger import logger
from CuriousRL.algorithm import AlgoWrapper
from CuriousRL.data import Data, Dataset
from .example_network import ThreeLayerAllConnetedNetwork, TwoLayerAllConnetedNetwork
import time as tm
import copy

class DQNWrapper(AlgoWrapper):
    """ This class is to build DQN algorithm.

    :param batch_size: bathch size
    :type batch_size: int
    :param lr: learning rate
    :type lr: float
    :param epsilon_greedy: epsilon parameter for greedy policy
    :type epsilon_greedy: float
    :param gamma_discount: discount factor for reward function
    :type gamma_discount: float
    :param target_replace_iter: target update frequency
    :type target_replace_iter: int
    :param buffer_size: maximum memory capacity
    :type buffer_size: int
    :param max_episode: number of time steps
    :type max_episode: int 
    """
    def __init__(self, 
                batch_size=128, 
                lr=0.001, 
                epsilon_greedy=0.9, 
                gamma_discount=0.999, 
                target_replace_iter=10, 
                buffer_size=10000, 
                max_episode=1e3,
                network_module = None,
                optimizer_class = None):
        super().__init__(batch_size=int(batch_size), 
                         lr=lr, 
                         epsilon_greedy=epsilon_greedy, 
                         gamma_discount=gamma_discount, 
                         target_replace_iter=target_replace_iter, 
                         buffer_size=buffer_size,
                         max_episode=int(max_episode),
                         network_module = network_module,
                         optimizer_class = optimizer_class)
        self._step_counter = 0                                     # for target updating
        self._network_module = network_module
        self._optimizer_class = optimizer_class

        EPS_START = 1
        EPS_END = 0.05
        self.EPS_DECAY_LENGTH = int(1e4)
        self.EPS_LINEAR_STEP = (EPS_START-EPS_END)/self.EPS_DECAY_LENGTH
        self.EPS_DECAY_EXP = 0.9999
        self._eps_threshould = EPS_START

    def init(self, scenario: ScenarioWrapper) -> DQNWrapper:
        if self._network_module == None:
            if len(scenario.state_shape) > 1:
                pass # TODO use CNN  
            else:
                self._network_module = TwoLayerAllConnetedNetwork
        self._scenario = scenario
        self._dataset = Dataset(buffer_size=self.kwargs['buffer_size'])
        # TODO ALL DISCRETE # FIXME
        # TODO ALL CONTINUOUS
        # TODO SOME DISCRETE AND SOME CONTINUOUS 不要出现list，全部用tensor 或者numpy！！！！
        self._eval_net = self._network_module(self._scenario.current_state.shape[0], len(self._scenario.action_space._action_range[0]))
        self._target_net = copy.deepcopy(self._eval_net)
        if global_config.is_cuda:
            self._eval_net = self._eval_net.cuda()
            self._target_net = self._target_net.cuda()
        if self._optimizer_class == None:
            self._optimizer = torch.optim.SGD(self._eval_net.parameters(), lr = self.kwargs['lr'], weight_decay=1e-4)
        else:
            self._optimizer = self._optimizer_class(self._eval_net.parameters(), lr=self.kwargs['lr'])
        self._loss_func = nn.MSELoss()
        self._eval_net.eval()
        self._target_net.eval()
        return self

    def _choose_action(self, x):
        if self._step_counter < self.EPS_DECAY_LENGTH:
            self._eps_threshould -= self.EPS_LINEAR_STEP
        elif self._step_counter == self.EPS_DECAY_LENGTH:
            logger.info("[+] ------------- EXP MODE START ---------------------")
        else:
            self._eps_threshould *= self.EPS_DECAY_EXP
        if np.random.random() > self._eps_threshould:
            action = self._eval_net(x.unsqueeze(0)).max(1)[1].item()
        else:
            action = self._scenario.action_space.sample()
        return action
    
    def _learn(self, epi):
        # target parameter update
        self._eval_net.train()
        self._target_net.train()
        if epi % self.kwargs['target_replace_iter'] == 0:
            self._target_net.load_state_dict(self._eval_net.state_dict())
        sample_data = self._dataset.fetch_random_data(self.kwargs['batch_size'])
        q_eval = self._eval_net(sample_data.state).gather(1, sample_data.action.long().unsqueeze(1))
        with torch.no_grad():
            q_next = self._target_net(sample_data.next_state)
            q_target = sample_data.reward + (~sample_data.done_flag) * self.kwargs['gamma_discount'] * q_next.max(1)[0]
        loss = self._loss_func(q_eval, q_target.unsqueeze(1))
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._eval_net.eval()
        self._target_net.eval()
    def solve(self):
        for epi in range(self.kwargs['max_episode']):
            current_state = torch.from_numpy(self._scenario.reset()).float().cuda()
            episode_reward = 0.
            while True:
                action = self._choose_action(current_state)
                new_data = self._scenario.step(action) # take action
                current_state = new_data.next_state
                episode_reward += new_data.reward
                self._scenario.render()
                self._dataset.update(new_data)
                if self._dataset.current_buffer_size > self.kwargs['batch_size']:
                    self._learn(epi)
                self._step_counter += 1
                if new_data.done_flag == True:
                    break
            logger.info('[+ +] Episode:%5d'%(epi) + '\t Episode Reward:%5d'%(episode_reward) + '\t Epsilon:%.5f'%(self._eps_threshould))
