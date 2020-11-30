
#%%
from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
                                            VehicleTracking, CarParking,\
                                            RoboticArmTracking, TwoLinkPlanarManipulator, \
                                            ThreeLinkPlanarManipulator
from CuriousRL.utils.Logger import logger

if __name__ == "__main__":
    logger.set_folder_name("test2", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
    scenario = RoboticArmTracking()
    BasiciLQR(max_iter=100, line_search_method="vanilla", stopping_method="relative").init(scenario).solve()
    scenario.play()
# %%
# from CuriousRL.data.data import Data
# from CuriousRL.data.dataset import Dataset
# import torch
# from CuriousRL.utils.config import global_config
# import time
# if __name__ == "__main__":
#     dataset_test3 = Dataset(10000, (5,5), 5, is_use_gpu=False)
#     time1 = time.time()
#     for i in range(10):
#         data_test1 = Data(torch.rand(5, 5, 5), torch.rand(5, 5), torch.rand(5), torch.ones(5,dtype=bool))
#         data_test2 = Data(torch.rand(5,5), torch.rand(5), 2., False)
#         data_test3 = data_test2.cat(data_test1)
#         dataset_test3.update_dataset(data_test3.cuda())
#     time2 = time.time()
#     print(time2 - time1)

# %%
