from __future__ import annotations
from typing import Union
import torch
from .data import Data
from .batch import Batch
from .data_parant import DataParant, ACCESSIBLE_KEY
from typing import Union

class Dataset(DataParant):
    """This is a class for building the dataset in the learning process. 
    The dimension of state and dimension of action are not necessary,
    because these infomation can be obtained when the dataset is updated.
    The only necessary parameter is **on_gpu** and **buffer_size**.

    :param on_gpu: Whether the dataset is saved as a GPU ``Tensor``.
    :type on_gpu: bool
    :param buffer_size: The size of the dataset. 
    :type buffer_size: int
    """

    def __init__(self, on_gpu: bool, buffer_size: int) -> None:
        self._on_gpu = on_gpu
        self._buffer_size = int(buffer_size)
        self._dataset_dict = {}
        self._update_index = 0
        self._total_update_num = 0
        self._random_batch_dict = {}
        self._random_batch_size = None

    def __len__(self):
        return self.current_buffer_size

    def __getitem__(self, index):
        temp_dict = {}
        for key in ACCESSIBLE_KEY:
            if self._dataset_dict[key] is not None:
                temp_dict[key] = self._dataset_dict[key][index]
        batch = Batch(**temp_dict, on_gpu=self.on_gpu)
        return batch

    def _init_dataset_from_data(self, data: Data):
        for key in ACCESSIBLE_KEY:
            if data._data_dict[key] is None:
                self._dataset_dict[key] = None
                continue
            if key in {"state", "next_state", "action"}:
                self._dataset_dict[key] = torch.zeros(
                    (self._buffer_size, *data._data_dict[key].shape), dtype=data._data_dict[key].dtype)
            elif key == "reward":
                self._dataset_dict[key] = torch.zeros(
                    (self._buffer_size), dtype=data._data_dict[key].dtype)
            elif key == "done_flag":
                self._dataset_dict[key] = torch.zeros(
                    (self._buffer_size), dtype=torch.bool)
            if self.on_gpu:
                self._dataset_dict[key] = self._dataset_dict[key].cuda()
            else:
                self._dataset_dict[key] = self._dataset_dict[key].cpu()

    def _init_dataset_with_batch(self, batch: Batch):
        for key in ACCESSIBLE_KEY:
            if batch._batch_dict[key] is None:
                self._dataset_dict[key] = None
                continue
            if key in {"state", "next_state", "action"}:
                self._dataset_dict[key] = torch.zeros(
                    (self._buffer_size, *batch._batch_dict[key].shape[1:]), dtype=batch._batch_dict[key].dtype)
            elif key == "reward":
                self._dataset_dict[key] = torch.zeros(
                    (self._buffer_size), dtype=batch._batch_dict[key].dtype)
            elif key == "done_flag":
                self._dataset_dict[key] = torch.zeros(
                    (self._buffer_size), dtype=torch.bool)
            if self.on_gpu:
                self._dataset_dict[key] = self._dataset_dict[key].cuda()
            else:
                self._dataset_dict[key] = self._dataset_dict[key].cpu()

    @property
    def current_buffer_size(self):
        """Get the number of data in the current buffer.

        :return: [description]
        :rtype: [type]
        """
        return min(self._total_update_num, self._buffer_size)

    def update(self, new_data: Union[Data, Batch]) -> None:
        """Update the new data into the dataset. If the dataset is full, 
        then this method will remove the oldest data in the dataset,
        and update the new data alternatively.

        The parameter can be a ``Data`` instance, or a ``Batch`` instance.

        :param new_data: New data
        :type new_data: Union[Data, Batch]
        """
        if self._total_update_num == 0:  # If this is the first update, initial the dataset
            if isinstance(new_data, Data):
                self._init_dataset_from_data(new_data)
            elif isinstance(new_data, Batch):
                self._init_dataset_with_batch(new_data)
            else:
                raise Exception(
                    'Only support updating dataset from Data and Batch instances.')
            super().__init__(on_gpu=self._on_gpu, dictionary=self._dataset_dict)

        if isinstance(new_data, Batch):
            new_data_len = len(new_data)
            self._total_update_num += new_data_len
            # if not exceed the last data in the dataset
            if self._update_index+new_data_len <= self._buffer_size:
                for key in ACCESSIBLE_KEY:
                    if self._dataset_dict[key] is not None:
                        self._dataset_dict[key][self._update_index:self._update_index +
                                                new_data_len] = new_data._batch_dict[key]
                self._update_index += new_data_len
                if self._update_index == self._buffer_size:
                    self._update_index = 0
            else:  # if exceed
                exceed_number = new_data_len + self._update_index - self._buffer_size
                for key in ACCESSIBLE_KEY:
                    if self._dataset_dict[key] is not None:
                        self._dataset_dict[key][self._update_index:
                                                ] = new_data._batch_dict[key][:self._buffer_size-self._update_index]
                        self._dataset_dict[key][:exceed_number] = new_data._batch_dict[key][self._buffer_size-self._update_index:]
                self._update_index = exceed_number
        elif isinstance(new_data, Data):
            self._total_update_num += 1
            for key in ACCESSIBLE_KEY:
                if self._dataset_dict[key] is not None:
                    self._dataset_dict[key][self._update_index] = new_data._data_dict[key]
            self._update_index += 1
            if self._update_index == self._buffer_size:
                self._update_index = 0

    def fetch_all_data(self) -> Batch:
        """Return all the data in the dataset.

        :return: All data
        :rtype: Batch
        """
        index = list(range(self.current_buffer_size))
        return self[index]

    def fetch_random_data(self, num_of_data: int) -> Batch:
        """Return the data with random index

        :param num_of_data: How many data will be returned
        :type num_of_data: int
        :return: Data with random keyes
        :rtype: Data
        """
        if self._random_batch_size != num_of_data:  # initial random batch
            self._random_batch_size = num_of_data
            for key in ACCESSIBLE_KEY:
                if self._dataset_dict[key] is None:
                    self._random_batch_dict[key] = None
                    continue
                if key in {"state", "next_state", "action"}:
                    self._random_batch_dict[key] = torch.zeros((num_of_data, *self._dataset_dict[key].shape[1:]), dtype=self._dataset_dict[key].dtype) 
                elif key == "reward":
                    self._random_batch_dict[key] = torch.zeros((num_of_data), dtype=self._dataset_dict[key].dtype)
                elif key == "done_flag":
                    self._random_batch_dict[key] = torch.zeros((num_of_data), dtype=torch.bool)
                if self.on_gpu:
                    self._random_batch_dict[key] = self._random_batch_dict[key].cuda()
                else:
                    self._random_batch_dict[key] = self._random_batch_dict[key].cpu()
        if num_of_data > self.current_buffer_size:
            raise Exception("The current buffer size %d < Number of random data size %d." % (
                self.current_buffer_size, num_of_data))
        if self.on_gpu:
            index = torch.randint(high=self.current_buffer_size, size=(num_of_data,)).cuda()
        else:
            index = torch.randint(high=self.current_buffer_size, size=(num_of_data,))
        
        for key in ACCESSIBLE_KEY:
            if self._dataset_dict[key] is not None:
                self._random_batch_dict[key].copy_(self._dataset_dict[key][index])
        return Batch(**self._random_batch_dict, on_gpu=self.on_gpu)