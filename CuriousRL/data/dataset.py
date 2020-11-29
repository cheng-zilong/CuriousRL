from __future__ import annotations
from typing import Union
import torch
import numpy as np
from .data import Data


class Dataset(object):
    """This is a class for building the dataset in the learning process.

    :param buffer_size: The size of the dataset. 
    :type buffer_size: int
    :param obs_dim: The dimension of the observation data. For example, 
        the observation dimension of a binary image 
        is (512, 512). The observation of a cartpole system is (4,) or 4, 
        which includes the position, velocity, angle of the
        pole, and the angular velocity of the pole.
    :type obs_dim: tuple(int) or int
    :param action_dim: The dimension of the action data. For example, the 
        observation dimension of a cartpole system is (1,) or 1, 
        which is the force applied to the cart.  the observation dimension 
        of a vehicle system is (2,) or 2, 
        which are the steering angle and the accelaration. 
    :type action_dim: tuple(int) or int
    :param is_use_gpu: Whether the dataset is saved on GPU. If True, the 
        dataset is saved on GPU, ortherwise, on CPU. If None,
        on GPU if GPU is avaliable and on CPU if GPU is unavaliable.
    :type is_use_gpu: bool or None, optional
    """

    def __init__(self, buffer_size,
                 obs_dim: Union[tuple(int), int],
                 action_dim: Union[tuple(int), int],
                 is_use_gpu=None):
        self._buffer_size = buffer_size
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        if is_use_gpu is None:
            if torch.cuda.is_available():
                self._is_use_gpu = True
            else:
                self._is_use_gpu = False
        else:
            self._is_use_gpu = is_use_gpu
        if isinstance(obs_dim, int):
            obs_dim = (obs_dim,)
        if isinstance(action_dim, int):
            action_dim = (action_dim,)
        if self._is_use_gpu:
            self._obs_set = torch.zeros((buffer_size, *obs_dim)).cuda()
            self._action_set = torch.zeros((buffer_size, *action_dim)).cuda()
            self._reward_set = torch.zeros((buffer_size)).cuda()
            self._done_flag_set = torch.zeros((buffer_size), dtype=torch.bool).cuda()
        else:
            self._obs_set = torch.zeros((buffer_size, *obs_dim))
            self._action_set = torch.zeros((buffer_size, *action_dim))
            self._reward_set = torch.zeros((buffer_size))
            self._done_flag_set = torch.zeros((buffer_size), dtype=torch.bool)
        self._update_index = 0
        self._total_update = 0  # totally number of obtained data

    def update_dataset(self, data: Data):
        """Update the new data into the dataset. If the dataset is full, 
        then this method will remove the oldest data in the dataset,
        and update the new data alternatively.

        :param data: New data
        :type data: Data
        """
        self._total_update += len(data)
        # if not exceed the last data in the dataset
        if self._update_index+len(data) <= self._buffer_size:
            self._obs_set[self._update_index:self._update_index +
                          len(data)] = data.get_obs()
            self._action_set[self._update_index:self._update_index +
                             len(data)] = data.get_action()
            self._reward_set[self._update_index:self._update_index +
                             len(data)] = data.get_reward()
            self._done_flag_set[self._update_index:self._update_index +
                                len(data)] = data.get_done_flag()
            self._update_index += len(data)
            if self._update_index == self._buffer_size:
                self._update_index = 0
        else:  # if exceed
            exceed_number = len(data) + self._update_index - self._buffer_size
            self._obs_set[self._update_index:] = data.get_obs(
            )[:self._buffer_size-self._update_index]
            self._action_set[self._update_index:] = data.get_action(
            )[:self._buffer_size-self._update_index]
            self._reward_set[self._update_index:] = data.get_reward(
            )[:self._buffer_size-self._update_index]
            self._done_flag_set[self._update_index:] = data.get_done_flag(
            )[0:self._buffer_size-self._update_index]
            ##########################
            self._obs_set[:exceed_number] = data.get_obs(
            )[self._buffer_size-self._update_index:]
            self._action_set[:exceed_number] = data.get_action(
            )[self._buffer_size-self._update_index:]
            self._reward_set[:exceed_number] = data.get_reward(
            )[self._buffer_size-self._update_index:]
            self._done_flag_set[:exceed_number] = data.get_done_flag(
            )[self._buffer_size-self._update_index:]
            self._update_index = exceed_number

    def get_current_buffer_size(self):
        """Get the current data buffer size. If the number of the current data is less than the buffer size, 
        return current number of data. Otherwise, return the size of the dataset.

        :return: [description]
        :rtype: [type]
        """
        return min(self._total_update, self._buffer_size)

    def fetch_all_data(self):
        """Return all the data in the dataset.

        :return: All data
        :rtype: Data
        """
        index = list(range(self._buffer_size))
        return self.fetch_data_by_index(index)

    def fetch_data_by_index(self, index: list) -> Data:
        """Return the data by specifying the index. For example, if index = [1,2,5], then three datas in the dataset will be returned. 

        :param index: The index of the data
        :type index: list
        :return: Specific data
        :rtype: Data
        """
        data = Data(self._obs_set[index], self._action_set[index],
                    self._reward_set[index], self._done_flag_set[index])
        return data

    def fetch_data_randomly(self, num_of_data: int) -> Data:
        """Return the data with random indexes

        :param num_of_data: How many data will be returned
        :type num_of_data: int
        :return: Data with random indexes
        :rtype: Data
        """
        if self._total_update < self._buffer_size:
            if num_of_data > self._total_update:
                raise Exception("The current buffer size is %d. Number of random data size is %d." % (self._total_update, num_of_data) +
                                "The latter must be less or equal than the former.")
            index = np.random.choice(
                self._total_update, size=num_of_data, replace=False)
        else:
            if num_of_data > self._buffer_size:
                raise Exception("The current buffer size is %d. Number of random data size is %d." % (self._buffer_size, num_of_data) +
                                "The latter must be less or equal than the former.")
            index = np.random.choice(
                self._buffer_size, size=num_of_data, replace=False)
        data = Data(self._obs_set[index], self._action_set[index],
                    self._reward_set[index], self._done_flag_set[index])
        return data

    def clone_to_cpu(self) -> Dataset:
        """Return a new dataset on cpu, with the same data in the current dataset.

        :return: The new dataset on cpu
        :rtype: Dataset
        """
        new_dataset_wrapper = Dataset(
            self._buffer_size, self._obs_dim, self._action_dim, is_use_gpu=False)
        new_dataset_wrapper.update_dataset(self.fetch_all_data())
        return new_dataset_wrapper

    def clone_to_gpu(self) -> Dataset:
        """Return a new dataset on gpu, with the same data in the current dataset.

        :return: The new dataset on gpu
        :rtype: Dataset
        """
        new_dataset_wrapper = Dataset(
            self._buffer_size, self._obs_dim, self._action_dim, is_use_gpu=True)
        new_dataset_wrapper.update_dataset(self.fetch_all_data())
        return new_dataset_wrapper
