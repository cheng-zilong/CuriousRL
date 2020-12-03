# %%
from __future__ import annotations
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import gym

from CuriousRL.utils.config import global_config
from CuriousRL.algorithm import AlgoWrapper

np.random.seed(1)
torch.manual_seed(1)


class SumTree(object):
    """
    This SumTree code is to store data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root
    
class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This class is to 
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        """
        build sumtree and the relavant parameters
        """
        self.tree = SumTree(capacity)

    def store(self, transition):
        """
        store data, and update sumtree
        """
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        """
        extract the sample
        """
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        """
        update the priority of sample in the sumtree after training. 
        """
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DQNPrioritizedReplay(AlgoWrapper):
    """This class is to build the DQN with prioritized experience replay
    """
    def __init__(self, 
                batch_size=32,
                learning_rate=0.005,
                epsilon_greedy_max=0.9,
                gamma_discount=0.9,
                target_replace_iter=500,
                memory_capacity=10000,                
                epsilon_greedy_increment=None,
                is_prioritized=True,
                is_output_graph=False,
                sess=None
    ):
        """
            Parameter
            -----------
            :param bathch_size: bathch size.
            :type bathch_size: int
            :param learning_rate: learning rate.
            :type learning_rate: float
            :param epsilon_greedy: epsilon parameter for greedy policy.
            :type epsilon_greedy: float
            :param gamma_discount: discount factor for reward function.
            :type gamma_discount: float
            :param target_replace_iter: target update frequency.
            :type target_replace_iter: int
            :param memory_capacity: maximum memory capacity.
            :type memory_capacity: int
            :param epsilon_greedy_increment: the increment rate of the epsilon in greedy policy.
            :type epsilon_greedy_increment: float
            :param is_prioritized: is to decide to use double q or not
            :type is_prioritized: boolean
            :param is_output_graph:
            :type is_output_graph: boolean
            :param num_episode: number of episode.
            :type num_episode: int 
            :param sess:
            :type sess:
        """
        super().__init__(batch_size=batch_size,
                         learning_rate=learning_rate,
                         epsilon_greedy_max=epsilon_greedy_max,
                         gamma_discount=gamma_discount,
                         target_replace_iter=target_replace_iter,
                         memory_capacity=memory_capacity,                
                         epsilon_greedy_increment=epsilon_greedy_increment,
                         is_prioritized=is_prioritized,
                         is_output_graph=is_output_graph,
                         sess=sess)
        self.epsilon = 0 if self.kwargs['epsilon_greedy_increment'] is not None else self.kwargs['epsilon_greedy_max']
        self.learn_step_counter = 0


        # self._build_net()
        # target_params = target_net.named_parameters()
        # eval_params = eval_net.named_parameters()

# %%
