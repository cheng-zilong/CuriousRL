#%%
from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, \
                                            VehicleTracking, CarParking,\
                                            RoboticArmTracking, TwoLinkPlanarManipulator, \
                                            ThreeLinkPlanarManipulator

if __name__ == "__main__":
    scenario = RoboticArmTracking()
    algo = BasiciLQR()
    algo.init(scenario,is_save_json=False, is_use_logger= False).solve()
    scenario.play()

# %%
