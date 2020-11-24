#%%
from CuriousRL.algorithm import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, VehicleTracking, CarParking, RoboticArmTracking
from CuriousRL import ProblemBuilderClass

if __name__ == "__main__":
    scenario = RoboticArmTracking(is_with_constraints=True, x=1, y=-2)
    algo = LogBarrieriLQR(max_iter = 1000)
    ProblemBuilderClass(scenario , algo).learn()
    scenario.play()
# %%
