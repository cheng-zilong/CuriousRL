from __future__ import annotations
import torch
from typing import Union, Tuple

from .data import Data
from .data_parant import DataParant, ACCESSIBLE_KEY


class Batch(DataParant):
    """This is the class for building a batch of reinforcement learning data inlcuding 
    **state**, **action**, **next_state**, **reward**, and **done_flag**.
    This is the basic data class for all the algorithms in the ``CuriousRL`` package.
    Each ``Batch`` instance can contain none, one or several 
    **state(s)**, **action(s)**, **next_state(s)**, **reward(s)**, and **done_flag(s)**.

    There are two ways to initial a ``Batch`` class. 
    The first way initializes the ``Batch`` class from a tuple of ``Data`` instances. 
    The Second way initializes the ``Batch`` class directly from several Torch ``Tensor``s (or Numpy ``Array``s).

    When initialize from a tuple of ``Data`` instances, no key is required, e.g. my_batch = Batch(my_data1, my_data1, ....).

    When initialize directly from several ``Tensor``s (or Numpy ``Array``s), 
    keys (*state**, **action**, **next_state**, **reward**, and **done_flag**) are required
    e.g. my_batch = Batch(state = Tensor(...), action = Tensor(...), ....).

    .. note::
        ``Numpy.array`` is also supported in this class as arguments, which can be used as the alternative type of ``Tensor``.
        The ``Numpy.array`` will be converted to PyTorch ``Tensor``.

    .. note::
        **state**, **action**, **next_state**, **reward**, and **done_flag** should not be given if it is not necessary for the algorithm.

    .. note::
        When initialize directly from several ``Tensor``s, the first index of 
        **state(s)**, **action(s)**, **next_state(s)**, **reward(s)**, and **done_flag(s)**
        must be the index of data.

    :param on_gpu: Whether the batch is saved as a GPU ``Tensor``.
    :type on_gpu: bool
    """

    def __init__(self, on_gpu, *args: Data, **kwargs):
        self._batch_dict = {}
        self._on_gpu = on_gpu
        if len(args) != 0:
            for key in ACCESSIBLE_KEY:
                if args[0]._data_dict[key] is None:
                    self._batch_dict[key] = None
                    continue
                self._batch_dict[key] = torch.stack(
                    [data._data_dict[key] for data in args])
                if self._on_gpu:
                    self._batch_dict[key] = self._batch_dict[key].cuda()
                else:
                    self._batch_dict[key] = self._batch_dict[key].cpu()
            super().__init__(on_gpu=self._on_gpu, dictionary=self._batch_dict)
        else:
            super().__init__(on_gpu=self._on_gpu, dictionary=self._batch_dict, **kwargs)
            for key in ACCESSIBLE_KEY:
                if (key in {'reward', 'done_flag'}) and (self._batch_dict[key] is not None):
                    if self._batch_dict[key].dim() != 1:
                        raise Exception(
                            '\"' + key + '\" must be one dimension array of scalars!')

    def __len__(self):
        for key in ACCESSIBLE_KEY:
            if self._batch_dict[key] is not None:
                return self._batch_dict[key].shape[0]

    def __setitem__(self, index, data: Data):
        if not isinstance(index, int):
            raise Exception("Index must be an integer!")
        for key in ACCESSIBLE_KEY:
            if data._data_dict[key] is None:
                continue
            self._batch_dict[key][index] = data._data_dict[key]

    def __getitem__(self, index):
        data_dict = {}
        for key in ACCESSIBLE_KEY:
            if self._batch_dict[key] is None:
                continue
            data_dict[key] = self._batch_dict[key][index]
        return Data(**data_dict, on_gpu=self.on_gpu)

    def cat(self, batches: Tuple[Batch]):
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

    def share_memmory_(self) -> Batch:
        """Moves the underlying storage to shared memory. Since cuda tensor sharing is not supported
        in windows, on_gpu must be set False when running on the windows OS.
        """
        new_batch = self.clone()
        for key in ACCESSIBLE_KEY:
            if new_batch._batch_dict[key] is not None:
                new_batch._batch_dict[key] = new_batch._batch_dict[key].share_memory_(
                )
        return new_batch
