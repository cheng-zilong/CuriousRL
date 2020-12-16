import numpy as np
import torch

class GlobalConfiguration(object):
    """This is a class for the global configeration. 
    This is a singleton mode class.
    ``is_cuda`` is the property for deciding whether the gpu is utilized.
    ``set_random_see(seed)`` provides global seed for numpy, PyTorch, and etc.
    """
    def __init__(self):
        self._is_cuda = True
        self._device = torch.device("cuda:0")
        
    def __new__(cls):  
        """This class uses singleton mode
        """
        if not hasattr(cls, '_instance'):
            orig = super(GlobalConfiguration, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance

    @property
    def is_cuda(self):
        return self._is_cuda

    @property
    def device(self):
        return self._device    

    def set_random_seed(self, seed:bool):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.random_seed = seed 

    def set_is_cuda(self, is_cuda:bool, device = None):
        self._is_cuda = is_cuda
        if self._is_cuda:
            if device == None:
                self._device = torch.device("cuda:0")
            else:
                self._device = device
        else:
            if device == None:
                self._device = torch.device("cpu")
            else:
                self._device = device


global_config = GlobalConfiguration()