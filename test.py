#%%
from CuriousRL.algorithm import BasiciLQR
from CuriousRL.scenario.dynamic_model import CartPoleSwingUp1, CartPoleSwingUp2, VehicleTracking
from CuriousRL import ProblemBuilderClass

if __name__ == "__main__":
    scenario = CartPoleSwingUp1(is_with_constraints=True)
    algo= BasiciLQR(max_iter = 5000)
    ProblemBuilderClass(scenario , algo).learn()
    scenario.play()
# %%
