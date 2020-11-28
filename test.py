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
    LogBarrieriLQR(max_iter=100, line_search_method="vanilla", stopping_method="relative").init(scenario).solve()
    
# %%
from CuriousRL.data.data import Data
from CuriousRL.data.dataset import Dataset
import torch
from CuriousRL.utils.config import global_config
if __name__ == "__main__":
    data_test1 = Data(torch.rand(1,5,5).cuda(), torch.rand(1,5), torch.rand(1), torch.rand(1))
    data_test2 = Data(torch.rand(1,5,5).cuda(), torch.rand(1,5), torch.rand(1), torch.rand(1))
    data_test3 = data_test2.cat((data_test2, data_test1, data_test1, data_test1, data_test1, data_test1))
    dataset_test1 = Dataset(3, (5,5), 5, is_use_gpu=True)
    dataset_test1.update_dataset(data_test1)
    dataset_test2 = Dataset(3, (5,5), 5)
    dataset_test2.update_dataset(data_test2)
    
    dataset_test3 = Dataset(8, (5,5), 5, is_use_gpu=False)
    dataset_test3.update_dataset(data_test3)

    print(dataset_test3.clone_to_gpu().fetch_all_data())

# %%
