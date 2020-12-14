from __future__ import annotations
from collections import deque
import torch
import torch.nn as nn
import numpy as np
import time as tm
import copy

from . import example_network
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
                 iter=20,
                 batch_size=32,
                 eps_start=1,
                 eps_end=0.05,
                 eps_linear_decay_len=1e4,
                 eps_exp_decay_rate=0.999,
                 gamma=0.999,
                 target_replace_frames=1000,
                 buffer_size=1e4,
                 one_iter_max_frame=1e6,
                 network=None,
                 optimizer=None,
                 is_render=False,
                 log_per_frames=1000):
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
        self._one_iter_max_frame = one_iter_max_frame
        self._is_render = is_render
        self._iter = iter
        self._log_per_frames = log_per_frames
        logger.info(iter=iter,
                    batch_size=int(batch_size),
                    eps_start=eps_start,
                    eps_end=eps_end,
                    eps_linear_decay_len=int(eps_linear_decay_len),
                    eps_exp_decay_rate=eps_exp_decay_rate,
                    gamma=gamma,
                    target_replace_frames=int(target_replace_frames),
                    buffer_size=int(buffer_size),
                    one_iter_max_frame=int(one_iter_max_frame),
                    network=network,
                    optimizer=optimizer,
                    is_render=is_render,
                    log_per_frames=log_per_frames)

    def init(self, scenario: Scenario) -> DiscreteDQN:
        self._scenario = scenario
        reset_elem = scenario.reset().elem
        reset_state = reset_elem.next_state
        self._mode = "singleton" if isinstance(reset_elem, Data) else "multiples"
        if self._network is None:
            if self._mode == "singleton" and reset_state.ndim > 1:  # is image
                self._network = example_network.ThreeLayerConvolutionalNetwork(
                    reset_state.shape[0], len(scenario.action_space._action_range[0]))
                self._work_num=1
            elif self._mode == "multiples" and reset_state.ndim > 2: # is image
                self._network = example_network.ThreeLayerConvolutionalNetwork(
                    reset_state.shape[1], len(scenario.action_space[0]._action_range[0]))
                self._work_num=reset_state.shape[0]
            else:  # not image
                self._network = example_network.TwoLayerAllConnetedNetwork(
                    reset_state.shape[0], len(scenario.action_space._action_range[0]))
                self._work_num=1
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
            with torch.no_grad():
                if self._mode == "singleton":
                    action = [self._eval_net(x.unsqueeze(0)).max(1)[1].item()]
                elif self._mode == "multiples":
                    action = self._eval_net(x).max(1)[1].unsqueeze(1).tolist()
        else:
            if isinstance(self._scenario.action_space, list):
                num = len(self._scenario.action_space)
                action = []
                for i in range(num):
                    action += [self._scenario.action_space[i].sample()]
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
        return loss.item()

    def train(self):
        len_deque = int(self._log_per_frames/self._work_num)
        reward_deque = deque([],maxlen=len_deque)
        total_start_time = tm.time()
        current_frame_counter = 0
        while(True):
            if current_frame_counter > self._one_iter_max_frame:
                break
            current_frame_counter += self._work_num
            self._frames_counter += self._work_num
            one_frame_start_time = tm.time()
            if self._frames_counter < self._eps_linear_decay_len:
                self._eps -= (self._eps_linear_decay_step*self._work_num)
            elif self._frames_counter == self._eps_linear_decay_len:
                logger.info(
                    "[+] ------------- EXPONENTIAL MODE START ---------------------")
            else:
                self._eps *= self._eps_exp_decay_rate
            action = self._choose_action(self._scenario.elem.next_state)
            new_data = self._scenario.step(action).elem  # take action
            reward_deque.append(self._scenario.step(action).elem.reward)
            if self._is_render:
                self._scenario.render()
            self._dataset.update(new_data)
            if self._dataset.current_buffer_size > self._batch_size:
                self._learn()
            if isinstance(self._scenario.action_space, ActionSpace): # if it is the singleton case, check done flag
                if new_data.done_flag:
                    self._scenario.reset()
            one_frame_end_time = tm.time()
            if current_frame_counter%self._log_per_frames == 0:
                logger.info('[+ +]' +  
                            '\n[+ + +] Total Frames:%8d' % (self._frames_counter) +
                            '\n[+ + +] Current Frames:%5d' % (current_frame_counter) +
                            '\n[+ + +] Avg. Reward Per Worker:%s' % (str((torch.sum(torch.stack(list(reward_deque)),dim=0)/len_deque).tolist())) +
                            '\n[+ + +] Epsilon:%.5f' % (self._eps) +
                            '\n[+ + +] Frames/s:%.5f' % (self._work_num/(one_frame_end_time - one_frame_start_time)) + 
                            '\n[+ + +] Total Time:%.5f' % (one_frame_end_time - total_start_time))


    def solve(self):
        self.train()
        # total_start_time = tm.time()
        # for epi in range(self._one_iter_max_frame):_max_frame_max_frame_max_frame_max_frame_max_frame
        #     episode_reward = 0.
        #     epi_start_time = tm.time()
        #     while True:
        #         if self._frames_counter < self._eps_linear_decay_len:
        #             self._eps -= self._eps_linear_decay_step
        #         elif self._frames_counter == self._eps_linear_decay_len:
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
        #         self._frames_counter += 1
        #         if isinstance(self._scenario.action_space, ActionSpace): # if it is the singleton case, check done flag
        #             if new_data.done_flag:
        #                 self._scenario.reset()
        #                 break
        #     epi_end_time = tm.time()
        #     logger.info('[+ +]  Total Frames:%8d' % (self._frames_counter) +
        #                 '\t Episode:%5d' % (epi) +
        #                 '\t Episode Reward:%5d' % (episode_reward) +
        #                 '\t Epsilon:%.5f' % (self._eps) +
        #                 '\t Episode Time:%.5f' % (epi_end_time - epi_start_time) +
        #                 '\t Total Time:%.5f' % (epi_end_time - total_start_time))