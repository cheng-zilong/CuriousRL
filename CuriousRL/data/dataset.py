from __future__ import annotations
from typing import Union
import torch
import numpy as np
from .data import Data
from .batch import Batch
from .data import ACCESSIBLE_KEY
from typing import TYPE_CHECKING, Union
from CuriousRL.utils.config import global_config

class Dataset(object):
    """This is a class for building the dataset in the learning process. 
    The dimension of state and dimension of action are not necessary,
    because these infomation can be obtained when the dataset is updated.
    The only necessary parameter is ``buffer_size``.

    :param buffer_size: The size of the dataset. 
    :type buffer_size: int
    """
    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._dataset_dict = {}
        self._update_index = 0
        self._total_update_num = 0  # totally number of obtained data


    def _init_dataset_from_data(self, data: Data):
        for key in ACCESSIBLE_KEY:
            if data._data_dict[key] == None:
                self._dataset_dict[key] = None
                continue
            if global_config.is_cuda:
                if key in {"state", "next_state","action"}:
                    self._dataset_dict[key] = torch.zeros((self._buffer_size, *data._data_dict[key].shape)).cuda()
                elif key == "reward":
                    self._dataset_dict[key] = torch.zeros((self._buffer_size)).cuda()
                elif key == "done_flag":
                    self._dataset_dict[key] = torch.zeros((self._buffer_size), dtype=torch.bool).cuda()
            else:
                if key in {"state", "next_state","action"}:
                    self._dataset_dict[key] = torch.zeros((self._buffer_size, *data._data_dict[key].shape))
                elif key == "reward":
                    self._dataset_dict[key] = torch.zeros((self._buffer_size))
                elif key == "done_flag":
                    self._dataset_dict[key] = torch.zeros((self._buffer_size), dtype=torch.bool)

    def _init_dataset_with_batch(self, batch: Batch):
        for key in ACCESSIBLE_KEY:
            if batch._batch_dict[key] == None:
                self._dataset_dict[key] = None
                continue
            if global_config.is_cuda:
                if key in {"state", "next_state","action"}:
                    self._dataset_dict[key] = torch.zeros((self._buffer_size, *batch._batch_dict[key].shape[1:])).cuda()
                elif key == "reward":
                    self._dataset_dict[key] = torch.zeros((self._buffer_size)).cuda()
                elif key == "done_flag":
                    self._dataset_dict[key] = torch.zeros((self._buffer_size), dtype=torch.bool).cuda()
            else:
                if key in {"state", "next_state","action"}:
                    self._dataset_dict[key] = torch.zeros((self._buffer_size, *batch._batch_dict[key].shape[1:]))
                elif key == "reward":
                    self._dataset_dict[key] = torch.zeros((self._buffer_size))
                elif key == "done_flag":
                    self._dataset_dict[key] = torch.zeros((self._buffer_size), dtype=torch.bool)

    def update(self, new_data: Union[Data, Batch] ):
        """Update the new data into the dataset. If the dataset is full, 
        then this method will remove the oldest data in the dataset,
        and update the new data alternatively.

        The parameter can be a ``Data`` instance, or a ``Batch`` instance.

        :param new_data: New data
        :type new_data: Union[Data, Batch]
        """
        if self._total_update_num == 0: # If this is the first update, initial the dataset
            if isinstance(new_data, Data):
                self._init_dataset_from_data(new_data)
            elif isinstance(new_data, Batch):
                self._init_dataset_with_batch(new_data)
            else:
                raise Exception('Only support updating dataset from Data and Batch instances.')
        if isinstance(new_data, Batch):
            new_data_len = len(new_data)
            self._total_update_num += new_data_len
            # if not exceed the last data in the dataset
            if self._update_index+new_data_len <= self._buffer_size:
                for key in ACCESSIBLE_KEY:    
                    if self._dataset_dict[key] != None:
                        self._dataset_dict[key][self._update_index:self._update_index + new_data_len] = new_data._batch_dict[key]
                self._update_index += new_data_len
                if self._update_index == self._buffer_size:
                    self._update_index = 0
            else:  # if exceed
                exceed_number = new_data_len + self._update_index - self._buffer_size
                for key in ACCESSIBLE_KEY:
                    if self._dataset_dict[key] != None:
                        self._dataset_dict[key][self._update_index:] = new_data._batch_dict[key][:self._buffer_size-self._update_index]
                        self._dataset_dict[key][:exceed_number] = new_data._batch_dict[key][self._buffer_size-self._update_index:]
                self._update_index = exceed_number
        elif isinstance(new_data, Data):
            self._total_update_num += 1
            for key in ACCESSIBLE_KEY:    
                if self._dataset_dict[key] != None:
                    self._dataset_dict[key][self._update_index] = new_data._data_dict[key]
            self._update_index += 1
            if self._update_index == self._buffer_size:
                self._update_index = 0

    @property
    def current_buffer_size(self):
        """Get the number of data in the current buffer.

        :return: [description]
        :rtype: [type]
        """
        return min(self._total_update_num, self._buffer_size)

    def fetch_all_data(self) -> Batch:
        """Return all the data in the dataset.

        :return: All data
        :rtype: Batch
        """
        index = list(range(self._buffer_size))
        return self.fetch_data(index)

    def fetch_data(self, index: list) -> Batch:
        """Return the data by specifying the index. 
        For example, if index = [1,2,5], then a batch with three datas in the dataset will be returned. 

        :param index: The index of the data
        :type index: list
        :return: Specific data
        :rtype: Data
        """
        temp_dict = {}
        for key in ACCESSIBLE_KEY:
            if self._dataset_dict[key] != None:
                temp_dict[key] = self._dataset_dict[key][index]
        data = Batch(**temp_dict)
        return data

    def fetch_random_data(self, num_of_data: int) -> Batch:
        """Return the data with random index

        :param num_of_data: How many data will be returned
        :type num_of_data: int
        :return: Data with random keyes
        :rtype: Data
        """
        if self._total_update_num < self._buffer_size:
            if num_of_data > self._total_update_num:
                raise Exception("The current buffer size is %d. Number of random data size is %d." % (self._total_update_num, num_of_data) +
                                "The latter must be less or equal than the former.")
            index = np.random.choice(
                self._total_update_num, size=num_of_data, replace=False)
        else:
            if num_of_data > self._buffer_size:
                raise Exception("The current buffer size is %d. Number of random data size is %d." % (self._buffer_size, num_of_data) +
                                "The latter must be less or equal than the former.")
            index = np.random.choice(
                self._buffer_size, size=num_of_data, replace=False)
        return self.fetch_data(index)

    def __len__(self):
        return self._buffer_size

    def __str__(self):
        string = ""
        for key in ACCESSIBLE_KEY:
            string += key
            string += ":\n" + str(self._dataset_dict[key]) + "\n"
        return string

    def __repr__(self):
        return self.__str__()