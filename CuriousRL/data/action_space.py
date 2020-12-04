from __future__ import annotations
import torch
from torch import Tensor
from typing import Union, Tuple
from CuriousRL.utils.config import global_config
import copy
import numpy as np

class ActionSpace(object):
    """ This is an action space class to determine the space of actions.
    """

    def __init__(self, action_range, action_type, action_info):
        """There are three basic components in the action space.
        Range denotes the range of the action. If an action is continuous, then the corresponding 
        range is a list with the lower bound and the upper bound. For example, in the vehicle tracking example,
        the range of the steering angle is [-0.6, 0.6]. Range is given as a list with the first index as the 
        index of action.  If an action is discrete, then the corresponding range is a list with each possible actions.
        For example, in the gym breakout-v0, there are 4 discrete actions ['NOOP', 'FIRE', 'RIGHT', 'LEFT']. Then the range
        of the action is given as ['NOOP', 'FIRE', 'RIGHT', 'LEFT'].

        :param range: The range of the actions.
        :type range: List[Union[[upper_bound, lower_bound], List], ...]
        :param type: Whether the action is "Continuous" or "Discrete"
        :type type: List[str, ...]
        :param info: The meaning of the action.
        :type info: List[str, ...]
        """
        self._action_range = action_range
        self._action_type = action_type
        self._action_info = action_info

    def __len__(self):
        return len(self._action_range)

    def __str__(self):
        string = ""
        for i in range(len(self)):
            string += self._action_info[i] + "\t (" + self._action_type[i] + "): \t" + str(self._action_range[i]) + "\n"
        return string

    def samples(self, sample_number = 1) -> List[List]:
        """Generate more than one random action data. If the action is continuous, then the number is chosen from a uniform distribution
        given the lower bound and the upper bound. If the action is discrete, then an index in the discrete action space 
        is chosen randomly. For example, in the gym breakout-v0, there are 4 discrete actions ['NOOP', 'FIRE', 'RIGHT', 'LEFT'].
        Then the return of this methpd with the ``sample_number = 10`` may be [[0],[2],[3],[1],[0],[0],[2],[3],[1],[1]].

        :param sample_number: The size of actions, defaults to 1
        :type sample_number: int, optional
        :return: The generated actions in the form of a List. The first index is the number of data. The second
            index is the index of each data. 
        :rtype: List
        """
        samples = []
        for i in range(sample_number):
            samples.append([self.sample()])
        return samples

    def sample(self) -> List:
        """Generate one random action data. If the action is continuous, then the number is chosen from a uniform distribution
        given the lower bound and the upper bound. If the action is discrete, then an index in the discrete action space 
        is chosen randomly. For example, in the gym breakout-v0, there are 4 discrete actions ['NOOP', 'FIRE', 'RIGHT', 'LEFT'].
        Then the return of this methoe may be [0].

        :param sample_number: The size of actions, defaults to 1
        :type sample_number: int, optional
        :return: The generated actions in the form of a List. The first index is the number of data. The second
            index is the index of each data. 
        :rtype: List
        """
        sample = []
        for j in range(len(self)):
            if self._action_type[j] == "Continuous":
                sample.append(np.random.uniform(low = self._action_range[j][0], high = self._action_range[j][1]))
            elif self._action_type[j] == "Discrete":
                sample.append(np.random.randint(len(self._action_range[j])))
            else:
                raise Exception("Action type can only be \"Continuous\" or \"Discrete\".")
        return sample


