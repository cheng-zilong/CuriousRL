# %%
# from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
# from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
#                                             VehicleTracking, CarParking,\
#                                             RoboticArmTracking, TwoLinkPlanarManipulator, \
#                                             ThreeLinkPlanarManipulator
from CuriousRL.algorithm.value_based.dqn import DQN
from CuriousRL.utils.Logger import logger
from CuriousRL.data.data import Data
from CuriousRL.data.dataset import Dataset
import torch
from CuriousRL.utils.config import global_config
import numpy as np
import time

import gym
if __name__ == "__main__":
    global_config.set_is_cuda(True)
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    dqn = DQN()
    dqn.init(env)
    dqn.solve()

# %%