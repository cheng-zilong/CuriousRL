from CuriousRL.scenario import OpenAIGym
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
                                            VehicleTracking, CarParking,\
                                            RoboticArmTracking, TwoLinkPlanarManipulator, \
                                            ThreeLinkPlanarManipulator
from CuriousRL.utils.Logger import logger
from CuriousRL.utils.config import global_config
from CuriousRL.value import DQNWrapper
import matplotlib.pyplot as plt
import time as tm

if __name__ == "__main__":
    logger.set_folder_name('test').set_is_use_logger(False).set_is_save_json(False)
    logger.set_level(logger.INFO)
    global_config.set_is_cuda(True)
    scenario = OpenAIGym('CartPole-v1')
    algo = DQNWrapper()
    algo.init(scenario).solve()