from __future__ import annotations
import time as tm
from CuriousRL.utils.Logger import logger
from .ilqr_dynamic_model import iLQRDynamicModel
from .ilqr_obj_fun import iLQRObjectiveFunction
from .ilqr_wrapper import iLQRWrapper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CuriousRL.scenario.dynamic_model.dynamic_model import DynamicModelWrapper

class BasiciLQR(iLQRWrapper):
    def __init__(self, 
                max_iter = 1000, 
                is_check_stop = True, 
                stopping_criterion = 1e-6,
                max_line_search = 10, 
                gamma = 0.5):
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
                        line_search_method = "vanilla",
                        stopping_method = "relative")

    def init(self, scenario: DynamicModelWrapper, is_use_logger = True, logger_folder = None, is_save_json = True):
        if not scenario.with_model() or scenario.is_action_discrete() or scenario.is_output_image():
            raise Exception("Scenario \"" + scenario.__class__.__name__ + "\" cannot learn with LogBarrieriLQR")
        if is_use_logger:
            logger.logger_init(logger_folder, is_save_json)
        self.scenario = scenario
        # Parameters for the model
        self.n = self.scenario.get_n()
        self.m = self.scenario.get_m()
        self.T = self.scenario.get_T()
        # Initialize the dynamic_model and objective function
        self.dynamic_model = iLQRDynamicModel(dynamic_function = scenario.get_dynamic_function(), 
                                                x_u_var = scenario.get_x_u_var(), 
                                                constr = scenario.get_constr(),
                                                init_state = scenario.get_init_state(), 
                                                init_input = scenario.get_init_input(), 
                                                add_param_var = None, 
                                                add_param = None)
        self.obj_fun = iLQRObjectiveFunction(obj_fun = scenario.get_obj_fun(),
                                                x_u_var = scenario.get_x_u_var(),
                                                add_param_var = scenario.get_add_param_var(),
                                                add_param = scenario.get_add_param())

    def get_obj_fun(self) -> iLQRObjectiveFunction:
        return self.obj_fun

    def get_dynamic_model(self) -> iLQRDynamicModel:
        return self.dynamic_model

    def solve(self):
        """ Solve the problem with classical iLQR
        """
        # Initialize the trajectory, F_matrix, objective_function_value_last, C_matrix and c_vector
        self.print_params()
        self.trajectory = self.dynamic_model.eval_traj()
        self.F_matrix = self.dynamic_model.eval_grad_dynamic_model(self.trajectory)
        self.init_obj = self.obj_fun.eval_obj_fun(self.trajectory)
        self.obj_fun_value_last = self.init_obj
        self.c_vector = self.obj_fun.eval_grad_obj_fun(self.trajectory)
        self.C_matrix = self.obj_fun.eval_hessian_obj_fun(self.trajectory)
        logger.info("[+ +] Initial Obj.Val.: %.5e"%(self.get_obj_fun_value()))
        # Start iteration
        start_time = tm.time()
        for i in range(self.max_iter):
            if i == 1:  # skip the compiling time 
                start_time = tm.time()
            iter_start_time = tm.time()
            self.backward_pass()
            backward_time = tm.time()
            obj, isStop = self.forward_pass()
            forward_time = tm.time()
            logger.info("[+ +] Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e"%(
                        i,  backward_time-iter_start_time,forward_time-backward_time,obj))
            logger.save_to_json(trajectory = self.get_traj().tolist())
            if isStop and self.is_check_stop:
                break
        end_time = tm.time()
        logger.info("[+ +] Completed! All Time:%.5e"%(end_time-start_time))
    
