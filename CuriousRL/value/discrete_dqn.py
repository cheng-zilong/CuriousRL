from __future__ import annotations
from collections import deque
import torch
from torch._C import dtype
from torch.functional import Tensor
import torch.nn as nn
import numpy as np
import time as tm
import copy

from .example_network import ThreeLayerConvolutionalNetwork, TwoLayerAllConnetedNetwork
from .action_select import DiscreteActionSelect
from CuriousRL.scenario import Scenario
from CuriousRL.utils.config import global_config
from CuriousRL.utils.Logger import logger
from CuriousRL.algorithm import Algorithm
from CuriousRL.data import Data, Dataset
from CuriousRL.data.batch import Batch
from CuriousRL.data.action_space import ActionSpace

class DiscreteDQN(Algorithm):
    """ This class is to build DQN algorithm."""

    def __init__(self,
                 on_gpu,
                 iter_num=20,
                 batch_size=32,
                 eps_start=1,
                 eps_end=0.05,
                 eps_linear_decay_len=1e4,
                 eps_exp_decay_rate=0.999,
                 gamma=0.999,
                 target_replace_frames=1000,
                 buffer_size=10000,
                 train_frame_num=1e6,
                 test_num=10,
                 network=None,
                 optimizer=None,
                 is_train_render=False,
                 is_test_render=False,
                 log_per_frames=1000,
                 start_learning_frames = 10000):
        self._total_frames_counter = 0
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
        self._train_frame_num = train_frame_num
        self._test_num = test_num
        self._is_train_render = is_train_render
        self._is_test_render = is_test_render
        self._iter_num = iter_num
        self._log_per_frames = log_per_frames
        self._on_gpu = on_gpu
        self._start_learning_frames = start_learning_frames
        logger.info(on_gpu=on_gpu,
                    iter_num=iter_num,
                    batch_size=int(batch_size),
                    eps_start=eps_start,
                    eps_end=eps_end,
                    eps_linear_decay_len=int(eps_linear_decay_len),
                    eps_exp_decay_rate=eps_exp_decay_rate,
                    gamma=gamma,
                    target_replace_frames=int(target_replace_frames),
                    buffer_size=int(buffer_size),
                    one_iter_max_frame=int(train_frame_num),
                    test_num=int(test_num),
                    network=network,
                    optimizer=optimizer,
                    is_train_render=is_train_render,
                    is_test_render=is_test_render,
                    log_per_frames=log_per_frames,
                    start_learning_frames=start_learning_frames)

    def on_gpu(self):
        return self._on_gpu

    def init(self, scenario: Scenario) -> DiscreteDQN:
        self._scenario = scenario
        reset_elem = scenario.reset().elem
        reset_state = reset_elem.next_state
        if self._network is None:
            if self._scenario.mode == "single" and reset_state.ndim > 1:  # is image
                self._network = ThreeLayerConvolutionalNetwork(
                    reset_state.shape[0], len(scenario.action_space._action_range[0]))
                self._worker_num=1
            elif self._scenario.mode == "multiple" and reset_state.ndim > 2: # is image
                self._network = ThreeLayerConvolutionalNetwork(
                    reset_state.shape[1], len(scenario.action_space[0]._action_range[0]))
                self._worker_num=reset_state.shape[0]
            else:  # not image
                self._network = TwoLayerAllConnetedNetwork(
                    reset_state.shape[0], len(scenario.action_space._action_range[0]))
                self._worker_num=1
        self._dataset = Dataset(buffer_size=self._buffer_size, on_gpu=self._on_gpu)
        self._eval_net = self._network
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                params=self._eval_net.parameters(), lr=1e-4)
        self._target_net = copy.deepcopy(self._eval_net)
        if self._on_gpu:
            self._eval_net = self._eval_net.cuda()
            self._target_net = self._target_net.cuda()
        self._loss_func = nn.MSELoss()
        return self

    def _learn(self):
        if self._total_frames_counter % self._target_replace_frames == 0:
            self._target_net.load_state_dict(self._eval_net.state_dict())
        sample_data = self._dataset.fetch_random_data(self._batch_size)
        q_eval = self._eval_net(sample_data.state.float()).gather(
            1, sample_data.action.long())
        with torch.no_grad():
            q_next = self._target_net(sample_data.next_state.float())
            q_target = sample_data.reward + \
                (~sample_data.done_flag) * self._gamma * q_next.max(1)[0]
        loss = self._loss_func(q_eval, q_target.unsqueeze(1))
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()

    def train(self):
        self._scenario.reset()
        total_reward=0
        for current_frame_counter in range(0, self._train_frame_num, self._worker_num):
            one_frame_start_time = tm.time()
            if self._total_frames_counter < self._eps_linear_decay_len:
                self._eps -= (self._eps_linear_decay_step*self._worker_num)
            elif self._total_frames_counter == self._eps_linear_decay_len:
                logger.info(
                    "[+] ------------- EXPONENTIAL MODE START ---------------------")
            else:
                self._eps *= self._eps_exp_decay_rate
            action = DiscreteActionSelect.eps_greedy(net=self._network, scenario=self._scenario,eps=self._eps)
            new_elem = self._scenario.step(action).elem
            total_reward += torch.sum(new_elem.reward)
            if self._is_train_render:
                self._scenario.render()
            self._dataset.update(new_elem)
            if self._dataset.current_buffer_size > self._start_learning_frames:
                self._learn()
            if isinstance(self._scenario.action_space, ActionSpace): # if it is the single case, check done flag, in multiple case, auto reset
                if new_elem.done_flag:
                    self._scenario.reset()
            one_frame_end_time = tm.time()
            if current_frame_counter%self._log_per_frames == 0:
                logger.info('\n[+ +]' +  
                            '\n[+ + +] Total Frames:%d' % (self._total_frames_counter) +
                            '\n[+ + +] Current Frames:%d' % (current_frame_counter) +
                            '\n[+ + +] Total. Reward:%d' % (total_reward) +
                            '\n[+ + +] Epsilon:%.5f' % (self._eps) +
                            '\n[+ + +] Frames/s:%.5f' % (self._worker_num/(one_frame_end_time - one_frame_start_time)) + 
                            '\n[+ + +] Total Time:%.5f' % (one_frame_end_time - self._iter_start_time))
                total_reward = 0
            self._total_frames_counter += self._worker_num

    def test(self) -> Tensor:
        self._scenario.reset()
        episode_reward=torch.zeros(self._worker_num)
        done_flag_indicator = torch.zeros(self._worker_num, dtype=bool)
        if self._scenario.on_gpu:
            episode_reward = episode_reward.cuda()
            done_flag_indicator = done_flag_indicator.cuda()
        while(True):
            action = DiscreteActionSelect.eps_greedy(net=self._network, scenario=self._scenario, eps=0)
            new_elem = self._scenario.step(action).elem
            if self._is_test_render:
                self._scenario.render()
            if isinstance(self._scenario.action_space, ActionSpace): # if it is the single case, check done flag, in multiple case, auto reset
                if new_elem.done_flag:
                    self._scenario.reset()
            episode_reward[~done_flag_indicator] += new_elem.reward[~done_flag_indicator]
            done_flag_indicator = done_flag_indicator | new_elem.done_flag
            if all(done_flag_indicator):
                break
        logger.info('\n[* *]' +  
                    '\n[* * *] Total. Reward:%s' % str(episode_reward) +
                    '\n[* * *] Current Test Average Episode Reward:%.5f' % torch.mean(episode_reward))
        return torch.mean(episode_reward)
        
    def solve(self):
        for _ in range(self._iter_num):
            logger.info("[+] Start Training....")
            self._iter_start_time = tm.time()
            self.train()
            total_test_reward = 0
            for i in range(self._test_num):
                logger.info("[*] Start Test (%d)...."%(i))
                avg_episode_reward = self.test()
                total_test_reward += avg_episode_reward
            logger.info("[*] Average Episode Reward for All Test:%.5f"%(total_test_reward/self._test_num))
        # total_start_time = tm.time()
        # for epi in range(self._one_iter_max_frame):
        #     episode_reward = 0.
        #     epi_start_time = tm.time()
        #     while True:
        #         if self._total_frames_counter < self._eps_linear_decay_len:
        #             self._eps -= self._eps_linear_decay_step
        #         elif self._total_frames_counter == self._eps_linear_decay_len:
        #             logger.info(
        #                 "[+] ------------- EXPONENTIAL MODE START ---------------------")
        #         else:
        #             self._eps *= self._eps_exp_decay_rate
        #         action = self._choose_action(self._scenario.elem.next_state)
        #         new_data = self._scenario.step(action).elem  # take action
        #         episode_reward += new_data.reward
        #         if self._is_render:
        #             self._scenario.render()
        #         self._dataset.update(new_data)
        #         if self._dataset.current_buffer_size > self._batch_size:
        #             self._learn()
        #         self._total_frames_counter += 1
        #         if isinstance(self._scenario.action_space, ActionSpace): # if it is the single case, check done flag
        #             if new_data.done_flag:
        #                 self._scenario.reset()
        #                 break
        #     epi_end_time = tm.time()
        #     logger.info('[+ +]  Total Frames:%8d' % (self._total_frames_counter) +
        #                 '\n[+ + +]  Episode:%5d' % (epi) +
        #                 '\n[+ + +] Episode Reward:%5d' % (episode_reward) +
        #                 '\n[+ + +] Epsilon:%.5f' % (self._eps) +
        #                 '\n[+ + +] Episode Time:%.5f' % (epi_end_time - epi_start_time) +
        #                 '\n[+ + +] Total Time:%.5f' % (epi_end_time - total_start_time))