#%%
import numpy as np
import sympy as sp
from scipy import io
import time as tm
import os 
from numba import njit
from CuriousRL.utils.Logger import logger
from CuriousRL.algorithm.algo_wrapper import AlgoWrapper
from CuriousRL.scenario.dynamic_model.dynamic_model import DynamicModelWrapper
from .ilqr_dynamic_model import iLQRDynamicModel
from .basic_ilqr import iLQRWrapper
from .ilqr_obj_fun import iLQRObjectiveFunction

class LogBarrieriLQR(iLQRWrapper):
    def __init__(self, 
                max_iter = 1000, 
                is_check_stop = True, 
                stopping_criterion = 1e-6,
                max_line_search = 10, 
                gamma = 0.5,
                t = [0.5, 1., 2., 5., 10., 20., 50., 100., 500.]):
        """
            Parameter
            -----------
            max_iter : int
                Maximum number of iterations
            is_check_stop : boolean
                Whether the stopping criterion is checked
            stopping_criterion : double
                The stopping criterion
            max_line_search : int
                Maximum number of line search
            gamma : double 
                Gamma is the parameter for the line search: alpha=gamma*alpha

        """
        super().__init__(max_iter = max_iter, 
                        is_check_stop = is_check_stop, 
                        stopping_criterion = stopping_criterion,
                        max_line_search = max_line_search, 
                        gamma = gamma,
                        line_search_method = "feasibility",
                        stopping_method = "vanilla")
        self.t = t


    def init(self, scenario: DynamicModelWrapper):
        """ Initialize the iLQR solver class

            Parameter
            -----------
            dynamic_model : DynamicModelWrapper
                The dynamic model of the system
            obj_fun : ObjectiveFunctionWrapper
                The objective function of the iLQR
        """
        if not scenario.with_model() or scenario.is_action_discrete() or scenario.is_output_image():
            raise Exception("Scenario \"" + scenario.name + "\"cannot learn with LogBarrieriLQR")
        self.scenario = scenario
        # Parameters for the model
        self.n = self.scenario.get_n()
        self.m = self.scenario.get_m()
        self.T = self.scenario.get_T()
        # Initialize the dynamic_model and objective function
        self.constr = scenario.get_constr()
        self.dynamic_model = iLQRDynamicModel(dynamic_function = scenario.get_dynamic_function(), 
                                                x_u_var = scenario.get_x_u_var(), 
                                                constr = self.constr,
                                                init_state = scenario.get_init_state(), 
                                                init_input = scenario.get_init_input(), 
                                                add_param_var = None, 
                                                add_param = None)
        self.real_obj_fun = iLQRObjectiveFunction(obj_fun = scenario.get_obj_fun(),
                                                x_u_var = scenario.get_x_u_var(),
                                                add_param_var = scenario.get_add_param_var(),
                                                add_param = scenario.get_add_param())
        obj_fun = scenario.get_obj_fun()
        x_u_var = scenario.get_x_u_var()
        t_var = sp.symbols('t') # introduce the parameter for log barrier
        add_param_var = scenario.get_add_param_var()
        if add_param_var is None:
            add_param_var = (t_var,)
        else:
            add_param_var = (t_var, *add_param_var)
        # construct the barrier objective function
        barrier_obj_fun = obj_fun
        # add the inequality constraints to the objective function
        for i, c in enumerate(self.constr):
            if not np.isinf(c[0]):
                barrier_obj_fun += (-1/t_var)*sp.log(-(c[0] - x_u_var[i]))
            if not np.isinf(c[1]):
                barrier_obj_fun += (-1/t_var)*sp.log(-(x_u_var[i] - c[1]))
        if scenario.get_add_param() is None:
            add_param = self.t[0]*np.ones((self.T, 1), dtype = np.float64)
        else:
            add_param = np.hstack([self.t[0]*np.ones((self.T, 1)), scenario.get_add_param()])
        self.obj_fun = iLQRObjectiveFunction(   obj_fun = barrier_obj_fun,
                                                x_u_var = x_u_var,
                                                add_param_var = add_param_var,
                                                add_param = add_param)

        # Initialize the trajectory, F_matrix, objective_function_value_last, C_matrix and c_vector
        self.trajectory = self.dynamic_model.eval_traj()
        self.F_matrix = self.dynamic_model.eval_grad_dynamic_model(self.trajectory)
        self.init_obj = self.obj_fun.eval_obj_fun(self.trajectory)
        self.obj_fun_value_last = self.init_obj
        self.c_vector = self.obj_fun.eval_grad_obj_fun(self.trajectory)
        self.C_matrix = self.obj_fun.eval_hessian_obj_fun(self.trajectory)

    def clear_obj_fun_value_last(self):
        self.obj_fun_value_last = np.inf

    def solve(self):
        """ Solve the problem with classical iLQR
        """
        logger.info("[+ +] Initial Obj.Val.: %.5e"%(self.real_obj_fun.eval_obj_fun(self.get_traj())))
        for j in self.t:
            for i in range(self.max_iter):
                if i == 1:  # skip the compiling time 
                    start_time = tm.time()
                iter_start_time = tm.time()
                self.backward_pass()
                backward_time = tm.time()
                _, isStop = self.forward_pass(line_search = "feasibility")
                forward_time = tm.time()
                obj = self.real_obj_fun.eval_obj_fun(self.get_traj())
                logger.info("[+ +] Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e"%(
                            i,  backward_time-iter_start_time,forward_time-backward_time,obj))
                logger.info("[+ +] Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e"%(
                            i,  backward_time-iter_start_time,forward_time-backward_time,obj))
                logger.save_to_json(trajectory = self.get_traj().tolist())
                if isStop and self.is_check_stop:
                    self.clear_obj_fun_value_last()
                    logger.info("[+ +] Complete One Inner Loop! The log barrier parameter t is %.5f"%(j) + " in this iteration!")
                    logger.info("[+ +] Iteration No.\t Backward Time \t Forward Time \t Objective Value")
                    break
        end_time = tm.time()
        logger.info("[+ +] Completed! All Time:%.5e"%(end_time-start_time))


# %%
