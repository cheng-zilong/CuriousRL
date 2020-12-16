#%%
from CuriousRL.scenario import OpenAIGym, Scenario
from CuriousRL.scenario.dynamic_model import *
from CuriousRL.utils.Logger import logger
from CuriousRL.utils.config import global_config
from CuriousRL.value import DiscreteDQN
from CuriousRL.data import Dataset, Batch, Data
import torchvision as tv
import torch
import matplotlib.pyplot as plt
import time as tm
import gym
from CuriousRL.scenario.openai_gym.atari_wrapper import AtariScenarioWrapper, wrap_deepmind


def main():
    logger.set_folder_name('v3').set_is_use_logger(True).set_is_save_json(True)
    logger.set_level(logger.DEBUG)
    global_config.set_is_cuda(True)
    # scenario = OpenAIGym(gym.make("CartPole-v1"))
    # algo = DiscreteDQN()
    # algo.init(scenario)
    # algo.solve()
    scenario = AtariScenarioWrapper(OpenAIGym(wrap_deepmind(gym.make("BreakoutNoFrameskip-v4"))))
    algo = DiscreteDQN(eps_linear_decay_len = 100000, eps_start=1, eps_end=0.02, gamma=0.99, eps_exp_decay_rate = 1, max_episode=2000)
    algo.init(scenario)
    algo.solve()








    # print(scenario.reset().data)
    # print(scenario.step(scenario.action_space.sample()).data.next_state)
    # tv.utils.save_image(scenario.step(scenario.action_space.sample()).data.next_state[0], 'letmeseesee.png')
    # scenario = CartPoleSwingUp1()
# %load_ext line_profiler
# %lprun -f DiscreteDQN.solve -f DiscreteDQN._learn -f Data.__init__  main()

if __name__ == "__main__":
    main()

# %%
