#%%
from CuriousRL.algorithm import BasiciLQR
from CuriousRL.scenario.dynamic_model import VehicleTracking
from CuriousRL.utils.Logger import logger

if __name__ == "__main__":
    logger.set_save_json(False)
    logger_id = logger.logger_init("test1")
    algo1 = BasiciLQR()
    scen1 = VehicleTracking(algo = algo1)
    scen1.learn("test1")
    logger.logger_destroy()

# %%
