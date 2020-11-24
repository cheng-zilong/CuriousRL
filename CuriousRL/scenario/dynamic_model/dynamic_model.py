#%%
import numpy as np
import sympy as sp
from numba import njit
from CuriousRL.scenario.scen_wrapper import ScenarioWrapper
from CuriousRL.utils.Logger import logger

class DynamicModelWrapper(ScenarioWrapper):
    """ In this example, the cartpole system is static at 0, 0, heading to the postive direction of the y axis\\
        We hope the vechile can tracking the reference y=-10 with the velocity 8, and head to the right\\
        x0: angle, x1: angular velocity, x2: position, x3: velocity, x4: force
    """
    def __init__(self, 
                dynamic_function, 
                x_u_var, 
                constr,
                init_state, 
                init_input,
                obj_fun,
                add_param_var = None,
                add_param = None): 
        self.dynamic_function = dynamic_function
        self.x_u_var = x_u_var
        self.init_state = init_state
        self.init_input = init_input
        self.n = int(init_state.shape[0])
        self.m = int(len(x_u_var) - self.n)
        self.T = int(init_input.shape[0])
        self.obj_fun = obj_fun
        self.add_param_var = add_param_var
        self.add_param = add_param
        self.constr = constr

    def with_model(self):
        return True

    def is_action_discrete(self):
        return False

    def is_output_image(self):
        return False

    def get_dynamic_function(self):
        return self.dynamic_function

    def get_dynamic_function_lamdify(self):
        return self.dynamic_function_lamdify

    def get_x_u_var(self):
        return self.x_u_var

    def get_init_state(self):
        return self.init_state

    def get_init_input(self):
        return self.init_input

    def get_n(self):
        return self.n

    def get_m(self):
        return self.m

    def get_T(self):
        return self.T

    def get_obj_fun(self):
        return self.obj_fun

    def get_add_param_var(self):
        return self.add_param_var

    def get_add_param(self):
        return self.add_param 
    
    def get_constr(self):
        return self.constr
    