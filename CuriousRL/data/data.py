from __future__ import annotations
import torch
from torch import Tensor
from typing import Union, Tuple
from CuriousRL.utils.config import global_config
import copy
import numpy as np
ACCESSIBLE_KEY = {'state', 'action', 'next_state', 'reward', 'done_flag'}

class Data(object):
    """This is the class for building the reinforcement learning data inlcuding state, action, next_state, reward, and done_flag.
    This is the basic data class for all the algorithms in the ``CuriousRL`` package. 
    Each ``Data`` instance can contain one or none state, action, next state, reward, and done flag.

    To ensure the homogeneity of data representation, the reward and done_flag in the data will be forced to be a scalar. 

    .. note::
        ``Numpy.array`` is also supported in this class, which can be used as the alternative type of ``Tensor``.

    .. note::
        State, action, next_state, reward, done_flag should not be given if it is not necessary for the algorithm.

    :param state: State
    :type state: Union[Tensor, numpy.array, ...]
    :param action: Action
    :type action: Union[Tensor, numpy.array, ...]
    :param next_state: Next state
    :type next_state: Union[Tensor, numpy.array, ...]
    :param reward: Reward
    :type reward: Union[Tensor, numpy.array, ...]
    :param done_flag: The flag deciding whether one episode is done
    :type done_flag: Union[Tensor, numpy.array, ...]
    """
    def __init__(self, **kwargs):
        self._data_dict = {}
        for key in ACCESSIBLE_KEY:
            if key not in kwargs:
                self._data_dict[key] = None
                continue
            if not isinstance(kwargs[key], Tensor): 
                # if not Tensor, change it to Tensor first
                kwargs[key] = torch.from_numpy(np.asarray(kwargs[key]))
            if (kwargs[key].dtype != torch.float) and (kwargs[key].dtype != torch.bool): 
                # if not bool and bot int, transfer it to float
                kwargs[key] = kwargs[key].float() 
            if (global_config.is_cuda) and (kwargs[key].get_device() == -1):   
                # if GPU is used, and kwargs is not on GPU, transfer it to GPU
                kwargs[key] = kwargs[key].cuda()
            if key in {'reward', 'done_flag'}: 
                # if the key is reward or done_flag, ensure that it is a scalar
                if kwargs[key].dim() != 0:
                    raise Exception('\"' + key + '\" must be a scalar!')
            self._data_dict[key] = kwargs[key]
            
    def __str__(self):
        string = ""
        for key in ACCESSIBLE_KEY:
            string += key
            string += ":\n" + str(self._data_dict[key]) + "\n"
        return string

    def __repr__(self):
        return self.__str__()

    @property
    def state(self) -> Tensor:
        """Get state

        :return: state
        :rtype: Tensor[data_size, \*state_dim]
        """
        return self._data_dict['state']

    @property
    def action(self) -> Tensor:
        """Get action

        :return: action
        :rtype: Tensor[data_size, action_dim]
        """
        return self._data_dict['action']

    @property
    def next_state(self) -> Tensor:
        """Get the next state

        :return: next state
        :rtype: Tensor[data_size, \*state_dim]
        """
        return self._data_dict['next_state']
        

    @property
    def reward(self) -> Tensor:
        """Get reward

        :return: reward
        :rtype: Tensor[data_size]
        """
        return self._data_dict['reward']

    @property
    def done_flag(self) -> Tensor:
        """Get done flag

        :return: Done flag 
        :rtype: Tensor[data_size]
        """
        return self._data_dict['done_flag']

    def clone(self) -> Data:
        """Clone a new Data instance with the same content on the same device.

        :return: The new Data instance
        :rtype: Data
        """
        return copy.deepcopy(self)
