# %%
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

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
        self.fc1 = nn.Linear(n_states, mid_size)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(mid_size, n_states)
        self.out.weight.data.normal_(0, 0.1)   # initialization

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
        x = self.fc1(x)
        x = nn.functional.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(AlgoWrapper):
    """ This class is to build DQN algorithm.
    """
    def __init__(self, 
                bathch_size=32, 
                learning_rate=0.01, 
                epsilon_greedy=0.9, 
                gamma_discount=0.9, 
                target_replace_iter=100, 
                memory_capacity=2000):
        """
            Parameter
            -----------
            :param bathch_size: Maximum number of the iLQR iterations.
            :type bathch_size: int
            :param learning_rate: Decide whether the stopping criterion is checked.
                If is_check_stop = False, then the maximum number of the iLQR iterations will be reached.
            :type learning_rate: float
            :param epsilon_greedy: epsilon parameter for greedy policy
            :type epsilon_greedy: float
            :param gamma_discount: discount factor for reward function
            :type gamma_discount: float
            :param target_replace_iter: target update frequency
            :type target_replace_iter: int
            :param memory_capacity: maximum memory capacity
            :type memory_capacity: int
        """
        super().__init__(bathch_size=bathch_size, 
                         learning_rate=learning_rate, 
                         epsilon_greedy=epsilon_greedy, 
                         gamma_discount=gamma_discount, 
                         target_replace_iter=target_replace_iter, 
                         memory_capacity=memory_capacity)
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()
    
    def init(self, scenario: DynamicModelWrapper) -> DQN:
        self.n_states = scenario()  # TODO connect to the scenario
        self.n_actions = scenario() # TODO connect to the scenario
        self.memory = np.zeros((self.kwargs['memory_capacity'], self.n_states*2+2))     # initialize memory_capacity
        self.env_action_shape = 0          # TODO connect to the action space in this scenario

    
    def _choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.kwargs['epsilon_greedy']:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_action_shape == 0 else action.reshape(self.env_action_shape)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.n_states )
            action = action if self.env_action_shape == 0 else action.reshape(self.env_action_shape)
        return action
    
    def _store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.kwargs['memory_capacity']
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.kwargs['target_replace_iter'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.kwargs['memory_capacity'], self.kwargs['batch_size'])
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_actions+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.kwargs['gamma_discount'] * q_next.max(1)[0].view(self.kwargs['batch_size'], 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def solve(self):
        pass

# %%
aaa = DQN()
# %%
