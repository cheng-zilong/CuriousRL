
# #%%
# from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR, NNiLQR
# from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
#                                             VehicleTracking, CarParking,\
#                                             RoboticArmTracking, TwoLinkPlanarManipulator, \
#                                             ThreeLinkPlanarManipulator
# from CuriousRL.utils.Logger import logger
# import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     logger.set_folder_name("test2", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#     scenario = ThreeLinkPlanarManipulator()
#     # for i in range(1000):
#     #     scenario.step(scenario.action_space.generate_samples(1)[0])
#     #     scenario.render()
#     #     plt.gcf().canvas.set_window_title("Time:" + str(i))
#     #     plt.pause(0.001)
#     # print(scenario.action_space)
#     # NNiLQR(line_search_method="vanilla", stopping_method="relative", training_stopping_criterion=1e-3).init(scenario).solve()
#     # NNiLQR(gaussian_noise_sigma=[[0.01], [0.1]], iLQR_max_iter=10, training_stopping_criterion=1e-3).init(scenario).solve()
#     LogBarrieriLQR().init(scenario).solve()
#     scenario.play("test2",-1)

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
""