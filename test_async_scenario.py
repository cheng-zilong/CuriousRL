#%%
from CuriousRL.scenario import OpenAIGym, Scenario
from CuriousRL.scenario.scenario_async import ScenaroAsync
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
    logger.set_folder_name('v4').set_is_use_logger(True).set_is_save_json(True)
    logger.set_level(logger.DEBUG)
    global_config.set_is_cuda(True)

    # scenario = OpenAIGym(gym.make("CartPole-v1"))
    # algo = DiscreteDQN()
    # algo.init(scenario)
    # algo.solve()
    # scenario1 = AtariScenarioWrapper(OpenAIGym(wrap_deepmind("PongNoFrameskip-v4")))
    # scenario1.reset()
    # TODO WINDOWS on_gpu=True, not works
    scenario2 = ScenaroAsync(AtariScenarioWrapper(OpenAIGym(wrap_deepmind("PongNoFrameskip-v4"), on_gpu=False)), 10)
    scenario2.reset()

    # time1 = tm.time()
    # for i in range(1000):
    #     if i%10 == 0:
    #         print(i)
    #     a = scenario1.step(scenario1.action_space.sample()).elem
    # time2 = tm.time()
    # print(time2 -time1)

    # time1 = tm.time()
    # for i in range(1000):
    #     if i%10 == 0:
    #         print(i)
    #     action_list = []
    #     for _ in range(10):
    #         action_list += [scenario2.action_space[0].sample()]
    #     a = scenario2.step(action_list).elem
    # time2 = tm.time()
    # print(time2 -time1)

    algo = DiscreteDQN(on_gpu=True, eps_linear_decay_len = 1000000, eps_start=1, eps_end=0.02, gamma=0.99, eps_exp_decay_rate = 1, train_frame_num=100000, test_num=10, target_replace_frames=500, batch_size=64, buffer_size=100000, log_per_frames=10000)
    algo.init(scenario2)
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
