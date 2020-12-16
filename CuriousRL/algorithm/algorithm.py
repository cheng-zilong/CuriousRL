from abc import ABC, abstractmethod
from CuriousRL.utils.Logger import logger
from CuriousRL.scenario import Scenario
class Algorithm(ABC):
    """ This is a wrapper class for the algorithm implemetation.
        This is the basic class for all algorithms
        You can build your own algortihm based on this class
        An example is given as follows
            >>> class DIYalgorithm1(Algorithm): ...
            >>> algo1 = DIYalgorithm1(params) # init parameters. Call __init__(self, params)
            >>> scen1 = scenario("name", algo = algo1) # Ready for running. Call init(self, scenario)
            >>> scen1.learn() # Start learning. Call solve(self)
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.print_params()

    @abstractmethod
    def init(self, scenario:Scenario):
        pass
    
    @abstractmethod
    def solve(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__
    
    def print_params(self):
        """Save the parameters of the current scenario in logger.
        """
        logger.info("[+] Algorithm Name:" + str(self.name))
        for key in self.kwargs:
            try:
                logger.info("[+] " + key + " = " + str(self.kwargs[key].tolist()))
            except:
                logger.info("[+] " + key + " = " + str(self.kwargs[key]))

