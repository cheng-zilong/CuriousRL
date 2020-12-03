
#%%
from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
                                            VehicleTracking, CarParking,\
                                            RoboticArmTracking, TwoLinkPlanarManipulator, \
                                            ThreeLinkPlanarManipulator
from CuriousRL.utils.Logger import logger

# if __name__ == "__main__":
#     logger.set_folder_name("test2", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#     scenario = CartPoleSwingUp2()
#     BasiciLQR(max_iter=100, line_search_method="vanilla", stopping_method="relative").init(scenario).solve()
#     scenario.play()
# %%
from CuriousRL.data.data import Data
from CuriousRL.data.dataset import Dataset
import torch
from CuriousRL.utils.config import global_config
import numpy as np
import time
if __name__ == "__main__":
    dataset_test3 = Dataset(buffer_size = 5, state_dim = 5, action_dim= 2)
    global_config.set_is_cuda(False)
    time1 = time.time()
    data_test1 = Data(  state=np.random.random((3, 5)), 
                        action = np.random.random((3, 2)), 
                        reward = np.random.random(3), 
                        done_flag = np.random.choice(a=[False, True], size=(3)))
    data_test2 = Data(  state=np.random.random((3, 5)), 
                        action = np.random.random((3, 2)), 
                        reward = np.random.random(3), 
                        done_flag = np.random.choice(a=[False, True], size=(3)))
    data_test3 = data_test2.cat(data_test1)
    dataset_test3.update_dataset(data_test3)
    print(dataset_test3.fetch_all_data())
    dataset_test3.update_dataset(data_test3)
    print(dataset_test3.fetch_all_data())
    dataset_test3.update_dataset(data_test2)
    print(dataset_test3.fetch_all_data())
    dataset_test3.update_dataset(data_test2)
    print(dataset_test3.fetch_all_data())
    dataset_test3.update_dataset(data_test3)
    print(dataset_test3.fetch_all_data())
    time2 = time.time()
    print(time2 - time1)

# %%
