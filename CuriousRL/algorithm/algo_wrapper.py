from abc import ABC, abstractmethod
from CuriousRL.utils.Logger import logger

class AlgoWrapper(ABC):
    """ This is a wrapper class for the algorithm implemetation.
        This is the basic class for all algorithms
        You can build your own algortihm based on this class
        An example is given as follows
            >>> class DIYalgorithm1(AlgoWrapper): ...
            >>> algo1 = DIYalgorithm1(params) # init parameters. Call __init__(self, params)
            >>> scen1 = scenario("name", algo = algo1) # Ready for running. Call init(self, scenario)
            >>> scen1.learn() # Start learning. Call solve(self)
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.print_params()

    @abstractmethod
    def init(self, scenario):
        pass
    
    @abstractmethod
    def solve(self, is_use_logger, logger_folder, is_save_json):
        pass

    def print_params(self):
        for key in self.kwargs:
            logger.info("[+] " + key + " = " + str(self.kwargs[key]))
