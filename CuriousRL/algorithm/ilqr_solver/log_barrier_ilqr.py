from __future__ import annotations
import numpy as np
import sympy as sp
import time as tm
from CuriousRL.utils.Logger import logger
from .basic_ilqr import iLQRWrapper
from .ilqr_obj_fun import iLQRObjectiveFunction
from .ilqr_dynamic_model import iLQRDynamicModel
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CuriousRL.scenario.dynamic_model.dynamic_model import DynamicModelWrapper


class LogBarrieriLQR(iLQRWrapper):
    def __init__(self,
                 max_iter=1000,
                 is_check_stop=True,
                 stopping_criterion=1e-6,
                 max_line_search=50,
                 gamma=0.5,
                 t=[0.5, 1., 2., 5., 10., 20., 50., 100.],
                 line_search_method="feasibility",
                 stopping_method="relative"):
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
            line_search_method : str
            stopping_method : str
        """
        super().__init__(max_iter=max_iter,
                         is_check_stop=is_check_stop,
                         stopping_criterion=stopping_criterion,
                         max_line_search=max_line_search,
                         gamma=gamma,
                         line_search_method=line_search_method,
                         stopping_method=stopping_method)
        self.t = t

    def init(self, scenario: DynamicModelWrapper, is_use_logger=True, logger_folder=None, is_save_json=True) -> LogBarrieriLQR:
        """ Initialize the iLQR solver class

            Parameter
            -----------
            dynamic_model : DynamicModelWrapper
                The dynamic model of the system
            obj_fun : ObjectiveFunctionWrapper
                The objective function of the iLQR

            Return 
            LogBarrieriLQR
        """
        if not scenario.with_model() or scenario.is_action_discrete() or scenario.is_output_image():
            raise Exception("Scenario \"" + scenario.__class__.__name__ +
                            "\" cannot learn with LogBarrieriLQR")
        if is_use_logger:
            logger.logger_init(logger_folder, is_save_json)
        self.scenario = scenario
        # Parameters for the model
        self.n = self.scenario.get_n()
        self.m = self.scenario.get_m()
        self.T = self.scenario.get_T()
        self.constr = scenario.get_constr()
        self.dynamic_model = iLQRDynamicModel(dynamic_function=scenario.get_dynamic_function(),
                                              x_u_var=scenario.get_x_u_var(),
                                              constr=self.constr,
                                              init_state=scenario.get_init_state(),
                                              init_input=scenario.get_init_input(),
                                              add_param_var=None,
                                              add_param=None)
        self.real_obj_fun = iLQRObjectiveFunction(obj_fun=scenario.get_obj_fun(),
                                                  x_u_var=scenario.get_x_u_var(),
                                                  add_param_var=scenario.get_add_param_var(),
                                                  add_param=scenario.get_add_param())
        obj_fun = scenario.get_obj_fun()
        x_u_var = scenario.get_x_u_var()
        t_var = sp.symbols('t')  # introduce the parameter for log barrier
        add_param_var = scenario.get_add_param_var()
        if add_param_var is None:
            add_param_var = (t_var,)
        else:
            add_param_var = (*add_param_var, t_var)
        # construct the barrier objective function
        barrier_obj_fun = obj_fun
        # add the inequality constraints to the objective function
        for i, c in enumerate(self.constr):
            if not np.isinf(c[0]):
                barrier_obj_fun += (-1/t_var)*sp.log(-(c[0] - x_u_var[i]))
            if not np.isinf(c[1]):
                barrier_obj_fun += (-1/t_var)*sp.log(-(x_u_var[i] - c[1]))
        if scenario.get_add_param() is None:
            add_param = self.t[0]*np.ones((self.T, 1), dtype=np.float64)
        else:
            add_param = np.hstack(
                [scenario.get_add_param(), self.t[0]*np.ones((self.T, 1))])
        self.obj_fun = iLQRObjectiveFunction(obj_fun=barrier_obj_fun,
                                             x_u_var=x_u_var,
                                             add_param_var=add_param_var,
                                             add_param=add_param)
        return self

    def solve(self):
        """ Solve the problem with classical iLQR
        """
        # Initialize the trajectory, F_matrix, objective_function_value_last, C_matrix and c_vector
        self.print_params()
        self.trajectory = self.dynamic_model.eval_traj()
        self.F_matrix = self.dynamic_model.eval_grad_dynamic_model(
            self.trajectory)
        self.init_obj = self.obj_fun.eval_obj_fun(self.trajectory)
        self.obj_fun_value_last = self.init_obj
        self.c_vector = self.obj_fun.eval_grad_obj_fun(self.trajectory)
        self.C_matrix = self.obj_fun.eval_hessian_obj_fun(self.trajectory)
        # Start iteration
        logger.info("[+ +] Initial Obj.Val.: %.5e" %
                    (self.real_obj_fun.eval_obj_fun(self.get_traj())))
        for j in self.t:
            if j != self.t[0]:  # update t parameter
                add_param = self.get_obj_add_param()
                add_param[:, -1] = j*np.ones((self.T))
                self.set_obj_add_param(add_param)
            for i in range(self.max_iter):
                if j == self.t[0] and i == 1:  # skip the compiling time
                    start_time = tm.time()
                iter_start_time = tm.time()
                self.backward_pass()
                backward_time = tm.time()
                _, isStop = self.forward_pass()
                forward_time = tm.time()
                # do not care the value of log barrier
                obj = self.real_obj_fun.eval_obj_fun(self.get_traj())
                logger.info("[+ +] Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e" % (
                            i,  backward_time-iter_start_time, forward_time-backward_time, obj))
                logger.save_to_json(trajectory=self.get_traj().tolist())
                if isStop and self.is_check_stop:
                    self.set_obj_fun_value(np.inf)
                    logger.info(
                        "[+ +] Complete One Inner Loop! The log barrier parameter t is %.5f" % (j) + " in this iteration!")
                    break
        end_time = tm.time()
        logger.info("[+ +] Completed! All Time:%.5e" % (end_time-start_time))

    def get_obj_fun(self) -> iLQRObjectiveFunction:
        return self.obj_fun

    def get_dynamic_model(self) -> iLQRDynamicModel:
        return self.dynamic_model

    def generate_data(self):
        """Log barrier method does not use any data
        """
        pass

    def fetch_data(self):
        """Log barrier method does not use any data
        """
        pass
