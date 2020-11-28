import numpy as np
import torch



class GlobalConfiguration(object):
    def __new__(cls):  
        """This class uses singleton mode
        """
        if not hasattr(cls, '_instance'):
            orig = super(GlobalConfiguration, cls)
            cls._instance = orig.__new__(cls)
        return cls._instance

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.random_seed = seed 

global_config = GlobalConfiguration()