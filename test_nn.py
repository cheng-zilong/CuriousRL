#%%
from sympy.core import basic
from CuriousRL.algorithm.ilqr_solver import log_barrier_ilqr, nn_cilqr, nn_ilqr, basic_ilqr
from CuriousRL.scenario.vehicle_tracking import VehicleTracking, VehicleTrackingObs
from CuriousRL.scenario.car_parking import CarParking
from CuriousRL.scenario.quadcopter import QuadCopter
from CuriousRL.scenario.robotic_arm_tracking import RoboticArmTracking
from CuriousRL.utils.Logger import logger
import matplotlib.pyplot as plt

# if __name__ == "__main__":
# #     for i in range(1,50):
# #         try:
# #             logger.set_folder_name("QuadCopter_" + str(i), remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
# #             # scenario = VehicleTracking()
# #             scenario = QuadCopter() 
# #             # basic_ilqr.BasiciLQR().init(scenario).solve() 
# #             # log_barrier_ilqr.LogBarrieriLQR().init(scenario).solve() 
# #             # nn_ilqr.NNiLQR(gaussian_noise_sigma=[[0.01], [0.1]], iLQR_max_iter=100).init(scenario).solve() 
# #             nn_cilqr.NNiLQR(gaussian_noise_sigma=[[0.1], [0.1], [0.1], [0.1]], iLQR_max_iter=100).init(scenario).solve() 
# #         except Exception as e:
# #             pass
# #         continue
#     scenario = QuadCopter() 
#     scenario.play("QuadCopter_46")

# if __name__ == "__main__":
#     # for i in range(1,50):
#     #     try:
#     #         logger.set_folder_name("CarParking_" + str(i), remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
#     #         # scenario = VehicleTracking()
#     #         scenario = CarParking() 
#     #         # basic_ilqr.BasiciLQR().init(scenario).solve() 
#     #         # log_barrier_ilqr.LogBarrieriLQR().init(scenario).solve() 
#     #         # nn_ilqr.NNiLQR(gaussian_noise_sigma=[[0.01], [0.1]], iLQR_max_iter=100).init(scenario).solve() 
#     #         nn_cilqr.NNiLQR(gaussian_noise_sigma=[[0.01], [0.1]], iLQR_max_iter=100).init(scenario).solve() 
#     #     except Exception as e:
#     #         pass
#     #     continue
#     scenario = CarParking() 
#     scenario.play("CarParking_49")


if __name__ == "__main__":
    logger.set_folder_name("RoboticArmTracking_", remove_existing_folder=False).set_is_use_logger(True).set_is_save_json(True)
    # scenario = VehicleTracking()
    scenario = RoboticArmTracking() 
    # basic_ilqr.BasiciLQR().init(scenario).solve() 
    # log_barrier_ilqr.LogBarrieriLQR().init(scenario).solve() 
    # nn_ilqr.NNiLQR(gaussian_noise_sigma=[[0.01], [0.1]], iLQR_max_iter=100).init(scenario).solve() 
    nn_cilqr.NNiLQR(gaussian_noise_sigma=[[0.1], [0.1]], iLQR_max_iter=100, training_stopping_criterion=0.01, decay_rate_max_iters=200).init(scenario).solve() 
    # scenario = RoboticArmTracking() 
    scenario.play("RoboticArmTracking_")


# %%
