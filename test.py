#%%
from CuriousRL.algorithm import BasiciLQR
from CuriousRL.scenario.dynamic_model import VehicleTracking
from CuriousRL import ProblemBuilderClass

if __name__ == "__main__":
    ProblemBuilderClass(scenario = VehicleTracking() , algo= BasiciLQR()).learn()

# %%
