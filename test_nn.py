#%%
from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
                                            VehicleTracking, CarParking,\
                                            RoboticArmTracking, TwoLinkPlanarManipulator, \
                                            ThreeLinkPlanarManipulator
from CuriousRL.utils.Logger import logger

if __name__ == "__main__":
    logger.set_folder_name("test2").set_is_use_logger(True).set_is_save_json(True)
    scenario = ThreeLinkPlanarManipulator()
    LogBarrieriLQR(max_iter=100, line_search_method="vanilla", stopping_method="relative").init(scenario).solve()
    scenario.play()