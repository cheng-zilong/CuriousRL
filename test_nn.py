#%%
from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
from CuriousRL.algorithm.ilqr_solver.nn_ilqr import NNiLQR
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
                                            VehicleTracking, CarParking,\
                                            RoboticArmTracking, TwoLinkPlanarManipulator, \
                                            ThreeLinkPlanarManipulator
from CuriousRL.utils.Logger import logger

if __name__ == "__main__":
    logger.set_folder_name("test", remove_existing_folder=True).set_is_use_logger(True).set_is_save_json(False)
    scenario = RoboticArmTracking()
    NNiLQR().init(scenario).solve()
    scenario.play()