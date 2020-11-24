from CuriousRL.utils.Logger import logger
from CuriousRL.algorithm.algo_wrapper import AlgoWrapper
from CuriousRL.scenario.scen_wrapper import ScenarioWrapper
class ProblemBuilderClass(object):
    def __init__(self, scenario: ScenarioWrapper, algo: AlgoWrapper):
        """ Initialize the problem by specifying the scenario and the algorithm

            Parameter
            ---------
            scenario: ScenarioWrapper
                Specific scenario
            algo : AlgoWrapper
                Specific algorithm
        """
        self.algo = algo
        self.algo.init(scenario)

    def learn(self, is_use_logger = True, logger_folder = None, is_save_json = True):
        """ Start learning

            Parameter
            ---------
            is_use_logger: boolean
                Whether the result is saved in a log file
            logger_folder: string 
                if log is used, the folder name where the log file is saved
                if logger_folder = None, then the folder name is specified as the current time
            is_save_json: boolean
                Whether the results are saved in a json file
        """
        if is_use_logger:
            logger.logger_init(logger_folder, is_save_json)
        self.algo.print_params()
        self.algo.solve()
        if is_use_logger:
            logger.logger_destroy()