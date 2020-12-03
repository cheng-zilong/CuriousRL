# %%
from __future__ import annotations
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gym

from CuriousRL.utils.config import global_config
from CuriousRL.algorithm import AlgoWrapper

class Net(nn.Module):
    """ This class is to build a neural network for DQN.
    """
    def __init__(self, n_states, n_actions, mid_size=50):
        """ 
            Parameter
            -----------
            :param n_states: input feature size.
            :type n_states: int
            :param n_actions: output feature size.
            :type n_actions: int
        """
        super(Net, self).__init__()
        # Create a DQN model using CUDA
        self.device = torch.device("cuda" if global_config.set_is_cuda else "cpu")
        self.layers = nn.Sequential(
            nn.Linear(n_states, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, n_actions)
        ).to(self.device)

    def forward(self, x):
        """ 
            Parameter
            -----------
            :param x: input feature size.
            :type x: int

            Return
            -----------
            actions_value
        """
        return self.layers(x)
        

class DQN(AlgoWrapper):
    """ This class is to build DQN algorithm.
    """
    def __init__(self, 
                bathch_size=32, 
                learning_rate=0.01, 
                epsilon_greedy=0.9, 
                gamma_discount=0.9, 
                target_replace_iter=100, 
                memory_capacity=2000, 
                num_episode=400):
        """
            Parameter
            -----------
            :param bathch_size: bathch size
            :type bathch_size: int
            :param learning_rate: learning rate
            :type learning_rate: float
            :param epsilon_greedy: epsilon parameter for greedy policy
            :type epsilon_greedy: float
            :param gamma_discount: discount factor for reward function
            :type gamma_discount: float
            :param target_replace_iter: target update frequency
            :type target_replace_iter: int
            :param memory_capacity: maximum memory capacity
            :type memory_capacity: int
            :param num_episode: number of episode
            :type num_episode: int 
        """
        super().__init__(bathch_size=bathch_size, 
                         learning_rate=learning_rate, 
                         epsilon_greedy=epsilon_greedy, 
                         gamma_discount=gamma_discount, 
                         target_replace_iter=target_replace_iter, 
                         memory_capacity=memory_capacity,
                         num_episode=num_episode )
        
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
    
    def init(self, scenario: DynamicModelWrapper) -> DQN:
        self.n_states = scenario()  # TODO connect to the scenario
        self.n_actions = scenario() # TODO connect to the scenario
        self.memory = np.zeros((self.kwargs['memory_capacity'], self.n_states*2+2))     # initialize memory_capacity
        self.env_action_shape = 0          # TODO connect to the action space in this scenario

        self.eval_net, self.target_net = Net(self.n_states, self.n_actions), Net(self.n_states, self.n_actions)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()


    def _choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # input only one sample
        if np.random.uniform() < self.kwargs['epsilon_greedy']:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]     # return the argmax
        else:   # random
            action = np.random.randint(0, self.n_actions)
        return action
    
    def _store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))  # 2*n_states+2
        # replace the old memory with new memory
        index = self.memory_counter % self.kwargs['memory_capacity']
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def _learn(self):
        # target parameter update
        if self.learn_step_counter % self.kwargs['target_replace_iter'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.kwargs['memory_capacity'], self.kwargs['batch_size'])
        b_memory = self.memory[sample_index, :]
        b_state = Variable(torch.FloatTensor(b_memory[:, :self.n_states]))
        b_action = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_actions+1].astype(int)))
        b_reward = Variable(torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]))
        b_next_state = Variable(torch.FloatTensor(b_memory[:, -self.n_states:]))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_state).gather(1, b_action)  # shape (batch, 1)
        q_next = self.target_net(b_next_state).detach()     # detach from graph, don't backpropagate
        q_target = b_reward + self.kwargs['gamma_discount'] * q_next.max(1)[0].view(self.kwargs['batch_size'], 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def solve(self):
        for i_episode in range(self.kwargs['num_episode']):
            state = # TODO connect to scenario (initialization)
            episode_reward = 0
            while True:
                env.render()
                action = self._choose_action(state)

                # take action
                next_state, reward, is_check_stop, info = # TODO scenario(action) or env(action)

                # modify the reward
                x, x_dot, theta, theta_dot = s_
                r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                r = r1 + r2

                self._store_transition(state, action, reward, next_state)

                episode_reward += reward
                if self.memory_counter > self.kwargs['memory_capacity']:
                    self._learn()
                    if is_check_stop:
                        print('episode: ', i_episode, '| episode_reward: ', round(episode_reward, 2))

                if is_check_stop:
                    break
                state = next_state

# %%
# %%
