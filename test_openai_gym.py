from CuriousRL.scenario import OpenAIGym, Scenario
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
                                            VehicleTracking, CarParking,\
                                            RoboticArmTracking, TwoLinkPlanarManipulator, \
                                            ThreeLinkPlanarManipulator
from CuriousRL.utils.Logger import logger
from CuriousRL.utils.config import global_config
from CuriousRL.value import DiscreteDQN
import torchvision as tv
import torch
import matplotlib.pyplot as plt
import time as tm
import gym
from CuriousRL.scenario.openai_gym.atari_wrapper import AtariScenarioWrapper, wrap_deepmind


if __name__ == "__main__":
    logger.set_folder_name('test').set_is_use_logger(True).set_is_save_json(True)
    logger.set_level(logger.INFO)
    global_config.set_is_cuda(True)
    scenario = AtariScenarioWrapper(wrap_deepmind(gym.make("PongNoFrameskip-v4")))
    # scenario = CartPoleSwingUp1()
    algo = DiscreteDQN()
    algo.init(scenario)
    algo.solve()