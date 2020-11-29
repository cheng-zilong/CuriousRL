
# import matplotlib.patches as patches
# from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
# from CuriousRL.scenario.dynamic_model import ThreeLinkPlanarManipulator
# import numpy as np
# import matplotlib.pyplot as plt

# class ThreeLinkPlanarManipulatorDemo(object):
#     def __init__(self):
#         self.scenario = ThreeLinkPlanarManipulator()
#         fig, self.ax = self.scenario.create_plot(xlim=(-6,6), ylim=(-6,6))
#         self.algo = LogBarrieriLQR(max_line_search=10) # self.algo = LogBarrieriLQR(max_line_search = 10)
#         self.algo.init(self.scenario)
#         fig.canvas.mpl_connect('button_press_event', self.onclick)
#         self.algo.solve()
#         self.scenario.play()
#         plt.show()

#     def onclick(self, event):  
#         self.ax.patches = []
#         circle = patches.Circle((event.xdata, event.ydata), 0.1, alpha = 0.5)
#         circle.set_color('C4')
#         self.ax.add_patch(circle)
#         add_param = self.algo.get_obj_add_param()
#         add_param[:,0] = event.xdata
#         add_param[:,1] = event.ydata
#         self.algo.set_obj_add_param(add_param)
#         self.algo.set_obj_fun_value(np.inf)
#         self.algo.set_init_state(self.scenario.play_trajectory_current[0:self.scenario.get_n()].reshape(-1,1))
#         self.algo.solve()
#         self.scenario.play()

# if __name__ == "__main__":
#     ThreeLinkPlanarManipulatorDemo()

#%%
from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
                                            VehicleTracking, CarParking,\
                                            RoboticArmTracking, TwoLinkPlanarManipulator, \
                                            ThreeLinkPlanarManipulator
from CuriousRL.utils.Logger import logger

if __name__ == "__main__":
    logger.set_folder_name("test3").set_is_use_logger(True).set_is_save_json(True)
    scenario = ThreeLinkPlanarManipulator()
    BasiciLQR(max_iter=100, line_search_method="vanilla", stopping_method="relative").init(scenario).solve()
    scenario.play()
# %%
# from CuriousRL.data.data import Data
# from CuriousRL.data.dataset import Dataset
# import torch
# from CuriousRL.utils.config import global_config
# import time
# if __name__ == "__main__":
#     dataset_test3 = Dataset(10000, (5,5), 5, is_use_gpu=True)
#     time1 = time.time()
#     for i in range(10000):
#         data_test1 = Data(torch.rand(5, 5, 5), torch.rand(5, 5), torch.rand(5), torch.ones(5,dtype=bool))
#         # data_test2 = Data(torch.rand(5,5).cuda(), torch.rand(5).cuda(), 2., False)
#         # data_test3 = data_test2.cat(data_test1)
#         dataset_test3.update_dataset(data_test1)
#         dataset_test3.fetch_data_randomly(3)
#     time2 = time.time()
#     print(time2 - time1)

# %%
