from __future__ import annotations
from .data_parant import DataParant, ACCESSIBLE_KEY

class Data(DataParant):
    """This is the class for building the reinforcement learning data inlcuding
    **state**, **action**, **next_state**, **reward**, and **done_flag**.
    This is the basic data class for all the algorithms in the ``CuriousRL`` package. 
    Each ``Data`` instance can contain one or none **state**, **action**, **next_state**, **reward**, and **done_flag**.

    All data saved in the ``Data`` class are based on the PyTorch ``Tensor``.

    .. note::
        ``Numpy.array`` is also supported in this class as arguments, which can be used as the alternative type of ``Tensor``.
        The ``Numpy.array`` will be converted to PyTorch ``Tensor``.

    .. note::
        **state**, **action**, **next_state**, **reward**, and **done_flag** should not be given if it is not necessary for the algorithm.

    .. note:: 
        To ensure the homogeneity of data representation, **reward** and **done_flag** in the data will be forced to be a scalar. 

    :param on_gpu: Whether the data is saved as a GPU ``Tensor``.
    :type on_gpu: bool
    :param state: State
    :type state: Union[Tensor, numpy.array]
    :param action: Action
    :type action: Union[Tensor, numpy.array]
    :param next_state: Next state
    :type next_state: Union[Tensor, numpy.array]
    :param reward: Reward
    :type reward: Union[Tensor, numpy.array]
    :param done_flag: The flag deciding whether one episode is done
    :type done_flag: Union[Tensor, numpy.array]
    """
    def __init__(self, on_gpu:bool, **kwargs):
        self._data_dict = {}
        super().__init__(on_gpu=on_gpu, dictionary=self._data_dict, **kwargs)
        for key in ACCESSIBLE_KEY:
            if (key in {'reward', 'done_flag'}) and (self._data_dict[key] is not None):
                if self._data_dict[key].dim() != 0:
                    raise Exception('\"' + key + '\" must be a scalar!')