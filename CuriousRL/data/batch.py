from __future__ import annotations
import torch
from torch import Tensor
from typing import Union, Tuple
import copy
import numpy as np
from CuriousRL.utils.config import global_config
from .data import ACCESSIBLE_KEY, Data

class Batch(object):
    """This is the class for building a batch of reinforcement learning data inlcuding state, action, next_state, reward, and done_flag.
    This is the basic data class for all the algorithms in the ``CuriousRL`` package.
    Each ``Batch`` instance can contain none, one or several state(s), action(s), next state(s), reward(s), and done flag(s).

    There are two ways to initial a ``Batch`` class. 
    The first way initializes the ``Batch`` class from a tuple of ``Data`` instances. 
    The Second way initializes the ``Batch`` class directly from several tensor arrays (or numpy arrays).

    When initialize from a tuple of ``Data`` instances, no key is required, e.g. my_batch = Batch(my_data1, my_data1, ....).

    When initialize directly from several tensor arrays, keys (state, action, next_state, reward, done_flag) are required
    e.g. my_batch = Batch((state = tensor(...), action = tensor(...), ....)).

    .. note::
        ``Numpy.array`` is also supported in this class, which can be used as the alternative type of ``Tensor``.

    .. note::
        State, action, next_state, reward, done_flag should not be given if it is not necessary for the algorithm.

    .. note::
        When initialize directly from several tensor arrays, the first index if state, action, next state, reward, and done flag
        must be the number of data.

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

    def __init__(self, *args, **kwargs):
        self._batch_dict = {}
        if len(args) != 0:
            for key in ACCESSIBLE_KEY:
                if args[0]._data_dict[key] is None:
                    self._batch_dict[key] = None
                else:
                    self._batch_dict[key] = torch.stack(
                        [data._data_dict[key] for data in args])
        else:
            for key in ACCESSIBLE_KEY:
                if key not in kwargs:
                    self._batch_dict[key] = None
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
                if key in {'reward','done_flag'}:
                    # if the key is reward or done_flag, ensure that it is with one dimension
                    if kwargs[key].dim() != 1:
                        raise Exception('\"' + key + '\" must be one dimension array of scalars!')
                self._batch_dict[key] = kwargs[key]

    def __len__(self):
        for key in ACCESSIBLE_KEY:
            if self._batch_dict[key] is not None:
                return self._batch_dict[key].shape[0]

    def __str__(self):
        string = ""
        for key in ACCESSIBLE_KEY:
            string += key
            string += ":\n" + str(self._batch_dict[key]) + "\n"
        return string

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, index, data:Data):
        if not isinstance(index, int):
            raise Exception("Index must be an integer!")
        for key in ACCESSIBLE_KEY:
            if data._data_dict[key] is None:
                continue
            self._batch_dict[key][index] = data._data_dict[key]

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise Exception("Index must be an integer!")
        data_dict = {}
        for key in ACCESSIBLE_KEY:
            if data._data_dict[key] is None:
                data_dict[key] = None
            else:
                data_dict[key] = self._batch_dict[key][index]
        return Data(**data_dict)

    @property
    def state(self) -> Tensor:
        """Get state

        :return: state
        :rtype: Tensor[data_size, \*state_dim]
        """
        return self._batch_dict['state']

    @property
    def action(self) -> Tensor:
        """Get action

        :return: action
        :rtype: Tensor[data_size, action_dim]
        """
        return self._batch_dict['action']

    @property
    def next_state(self) -> Tensor:
        """Get the next state

        :return: next state
        :rtype: Tensor[data_size, \*state_dim]
        """
        return self._batch_dict['next_state']


    @property
    def reward(self) -> Tensor:
        """Get reward

        :return: reward
        :rtype: Tensor[data_size]
        """
        return self._batch_dict['reward']

    @property
    def done_flag(self) -> Tensor:
        """Get done flag

        :return: Done flag 
        :rtype: Tensor[data_size]
        """
        return self._batch_dict['done_flag']

    def cat(self, batches: Tuple[Batch, ...]):
        """Cat the current Batch instance with the other Batch instances. 
        The current batch will be updated. If you dont want to change the current one,
        the following method can be used. batch4 = batch1.clone().cat((batch2, batch3))

        :param batches: A tuple of batch with the same structure as the current one.
        :type batches: Tuple[Batch, ...]
        """
        if not isinstance(batches, (tuple, list)):
            batches = (self, batches)
        else:
            batches = (self, *batches)
        for key in ACCESSIBLE_KEY:
            if self._batch_dict[key] is not None:
                self._batch_dict[key] = torch.cat(
                    [batch._batch_dict[key] for batch in batches])
        return self

    def clone(self) -> Batch:
        """Clone a new Batch instance with the same content on the same device.

        :return: The new Batch instance
        :rtype: Batch
        """
        return copy.deepcopy(self)

    def share_memmory_(self, is_cpu = False) -> Batch:
        """Moves the underlying storage to shared memory. Since cuda tensor sharing is not supported
        in windows, is_cpu must be set true when running on the windows OS.
        """
        new_batch = self.clone()
        self.is_share_memmory_cpu = is_cpu
        if is_cpu:
            for key in ACCESSIBLE_KEY:
                if new_batch._batch_dict[key] is not None:
                    new_batch._batch_dict[key] = new_batch._batch_dict[key].cpu().share_memory_()
            return new_batch
        else:
            for key in ACCESSIBLE_KEY:
                if new_batch._batch_dict[key] is not None:
                    new_batch._batch_dict[key] = new_batch._batch_dict[key].share_memory_()
            return new_batch

    def to_gpu(self) -> Batch:
        """Moves the underlying storage to GPU"""
        new_batch = self.clone()
        for key in ACCESSIBLE_KEY:
            if new_batch._batch_dict[key] is not None:
                new_batch._batch_dict[key] = new_batch._batch_dict[key].cuda()
        return new_batch

    def to_cpu(self) -> Batch:
        """Moves the underlying storage to CPU"""
        new_batch = self.clone()
        for key in ACCESSIBLE_KEY:
            if new_batch._batch_dict[key] is not None:
                new_batch._batch_dict[key] = new_batch._batch_dict[key].cpu()
        return new_batch