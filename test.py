
#%%

from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR, NNiLQR
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
                                            VehicleTracking, CarParking,\
                                            RoboticArmTracking, TwoLinkPlanarManipulator, \
                                            ThreeLinkPlanarManipulator
from CuriousRL.utils.Logger import logger
import matplotlib.pyplot as plt

if __name__ == "__main__":
    logger.set_folder_name("test2", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
    scenario = ThreeLinkPlanarManipulator()
    # for i in range(1000):
    #     print (i)
    #     scenario.step(scenario.action_space.generate_samples(1)[0])
    #     scenario.render()
    #     plt.pause(0.001)
    # print(scenario.action_space)
    BasiciLQR(line_search_method="vanilla", stopping_method="relative").init(scenario).solve()
    # NNiLQR(gaussian_noise_sigma=[[0.01], [0.1]], iLQR_max_iter=10, training_stopping_criterion=1e-3).init(scenario).solve()
    scenario.play()
# # %%
# from CuriousRL.data.data import Data
# from CuriousRL.data.dataset import Dataset
# import torch
# from CuriousRL.utils.config import global_config
# import numpy as np
# import time
# if __name__ == "__main__":
#     dataset_test3 = Dataset(buffer_size = 5, state_dim = 5, action_dim= 2)
#     global_config.set_is_cuda(False)
#     time1 = time.time()
#     dataset1 = Dataset(buffer_size=100, state_dim=(512,512, 3), action_dim=10)
#     data1_from_gym = Data(state = np.random.random((512, 512, 3)), action = np.random.random(10), reward = 1)
#     data2_from_gym = Data(state = np.random.random((4, 512, 512, 3)), action = np.random.random((4, 10)), reward = np.random.random(4))
#     print(data1_from_gym)
#     print(data2_from_gym)
#     data3_from_gym = data1_from_gym.cat(data2_from_gym)
#     dataset1.update_dataset(data3_from_gym)

# # %%
# from CuriousRL.data.data import Data
# from CuriousRL.data.dataset import Dataset
# from CuriousRL.data.action_space import ActionSpace
# import torch
# from CuriousRL.utils.config import global_config
# import numpy as np
# import time
# if __name__ == "__main__":
#     a = ActionSpace([[-5, 5],['NOOP', 'FIRE', 'RIGHT', 'LEFT']], action_type=["Continuous", "Discrete"], action_info = ["First_action", "Second_action"])
#     print(a)
#     print(a.generate_samples(100))

# %%
