from __future__ import annotations
import torch
from torch import Tensor
from typing import Union, Tuple, TypeVar
from CuriousRL.utils.config import global_config
import copy
import numpy as np
ACCESSIBLE_KEY = {'state', 'action', 'next_state', 'reward', 'done_flag'}
TDataParant = TypeVar("TDataParant", bound="DataParant")

class DataParant(object):
    """This is the parant class for building the reinforcement learning data inlcuding
    **state**, **action**, **next_state**, **reward**, and **done_flag**.
    """
    def __init__(self, on_gpu:bool, dictionary:dict, **kwargs):
        self._dict = dictionary
        self._on_gpu=on_gpu
        for key in ACCESSIBLE_KEY:
            if key in dictionary:
                continue
            if key not in kwargs :
                self._dict[key] = None
                continue
            if kwargs[key] is None:
                self._dict[key] = None
                continue
            if not isinstance(kwargs[key], Tensor):
                kwargs[key] = torch.from_numpy(np.asarray(kwargs[key]))
            if on_gpu:
                kwargs[key] = kwargs[key].cuda()
            else:
                kwargs[key] = kwargs[key].cpu()
            self._dict[key] = kwargs[key]

    def __str__(self):
        string = ""
        for key in ACCESSIBLE_KEY:
            string += key
            string += ":\n" + str(self._dict[key]) + "\n"
        return string

    def __repr__(self):
        return self.__str__()

    @property
    def on_gpu(self):
        return self._on_gpu

    @property
    def state(self) -> Tensor:
        """Get state

        :return: state
        :rtype: Tensor[data_size, \*state_dim]
        """
        return self._dict['state']

    @property
    def action(self) -> Tensor:
        """Get action

        :return: action
        :rtype: Tensor[data_size, action_dim]
        """
        return self._dict['action']

    @property
    def next_state(self) -> Tensor:
        """Get the next state

        :return: next state
        :rtype: Tensor[data_size, \*state_dim]
        """
        return self._dict['next_state']

    @property
    def reward(self) -> Tensor:
        """Get reward

        :return: reward
        :rtype: Tensor[data_size]
        """
        return self._dict['reward']

    @property
    def done_flag(self) -> Tensor:
        """Get done flag

        :return: Done flag 
        :rtype: Tensor[data_size]
        """
        return self._dict['done_flag']

    def clone(self) -> TDataParant:
        """Clone a new Data instance with the same content on the same device.

        :return: The new Data instance
        :rtype: Data
        """
        return copy.deepcopy(self)

    def to_gpu(self) -> TDataParant:
        """Copy the underlying storage to GPU, and return a new DataParant. Self does not change."""
        new_data_parant = self.clone()
        for key in ACCESSIBLE_KEY:
            if new_data_parant._dict[key] is not None:
                new_data_parant._dict[key] = new_data_parant._dict[key].cuda()
        new_data_parant._on_gpu = True
        return new_data_parant

    def to_cpu(self) -> TDataParant:
        """Copy the underlying storage to CPU, and return a new DataParant. Self does not change."""
        new_data_parant = self.clone()
        for key in ACCESSIBLE_KEY:
            if new_data_parant._dict[key] is not None:
                new_data_parant._dict[key] = new_data_parant._dict[key].cpu()
        new_data_parant._on_gpu = False
        return new_data_parant