from __future__ import annotations
import numpy as np
import sympy as sp
import time as tm
from CuriousRL.utils.Logger import logger
from .basic_ilqr import iLQRWrapper
from .ilqr_obj_fun import iLQRObjectiveFunction
from .ilqr_dynamic_model import iLQRDynamicModel
from typing import TYPE_CHECKING
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
            :param max_iter: Maximum number of the iLQR iterations.
            :type max_iter: int
            :param is_check_stop: Decide whether the stopping criterion is checked.
                If is_check_stop = False, then the maximum number of the iLQR iterations will be reached.
            :type is_check_stop: bool
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
        super().__init__(stopping_criterion=stopping_criterion,
                         max_line_search=max_line_search,
                         gamma=gamma,
                         line_search_method=line_search_method,
                         stopping_method=stopping_method,
                         max_iter=max_iter,
                         is_check_stop=is_check_stop)
        self._t = t

    def init(self, scenario: DynamicModelWrapper) -> LogBarrieriLQR:
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
        if not isinstance(scenario, DynamicModelWrapper):
            raise Exception("Scenario \"" + scenario.name +
                            "\" cannot learn with LogBarrieriLQR")
        # Parameters for the model
        constr = scenario.constr
        self._dynamic_model = iLQRDynamicModel(dynamic_function=scenario.dynamic_function,
                                               xu_var=scenario.xu_var,
                                               constr=constr,
                                               init_state=scenario.init_state,
                                               init_action=np.zeros((scenario.T, scenario.m, 1)))
        self._real_obj_fun = iLQRObjectiveFunction(obj_fun=scenario.obj_fun,
                                                   xu_var=scenario.xu_var,
                                                   add_param_var=scenario.add_param_var,
                                                   add_param=scenario.add_param)
        xu_var = scenario.xu_var
        t_var = sp.symbols('t')  # introduce the parameter for log barrier
        add_param_var = scenario.add_param_var
        if add_param_var is None:
            add_param_var = (t_var,)
        else:
            add_param_var = (*add_param_var, t_var)
        # construct the barrier objective function
        barrier_obj_fun = scenario.obj_fun
        # add the inequality constraints to the objective function
        for i, c in enumerate(constr):
            if not np.isinf(c[0]):
                barrier_obj_fun += (-1/t_var)*sp.log(-(c[0] - xu_var[i]))
            if not np.isinf(c[1]):
                barrier_obj_fun += (-1/t_var)*sp.log(-(xu_var[i] - c[1]))
        if scenario.add_param is None:
            add_param = self._t[0] * \
                np.ones((self.dynamic_model._T, 1), dtype=np.float64)
        else:
            add_param = np.hstack(
                [scenario.add_param, self._t[0]*np.ones((self.dynamic_model._T, 1))])
        self._obj_fun = iLQRObjectiveFunction(obj_fun=barrier_obj_fun,
                                              xu_var=xu_var,
                                              add_param_var=add_param_var,
                                              add_param=add_param)
        return self

    def solve(self):
        """ Solve the problem with classical iLQR
        """
        # Initialize the trajectory, F_matrix, objective_function_value_last, C_matrix and c_vector
        self._trajectory = self._dynamic_model.eval_traj()  # init feasible trajectory
        C_matrix = self._obj_fun.eval_hessian_obj_fun(self._trajectory)
        c_vector = self._obj_fun.eval_grad_obj_fun(self._trajectory)
        F_matrix = self._dynamic_model.eval_grad_dynamic_model(
            self._trajectory)
        # Start iteration
        logger.info("[+ +] Initial Obj.Val.: %.5e" %
                    (self._real_obj_fun.eval_obj_fun(self._trajectory)))
        total_iter_no = -1
        for j in self._t:
            if j != self._t[0]:  # update t parameter
                add_param = self.get_obj_add_param()
                add_param[:, -1] = j*np.ones((self.dynamic_model._T))
                self.set_obj_add_param(add_param)
            for i in range(self.kwargs['max_iter']):
                total_iter_no += 1
                if j == self._t[0] and i == 1:  # skip the compiling time
                    start_time = tm.time()
                iter_start_time = tm.time()
                K_matrix, k_vector = self.backward_pass(
                    C_matrix, c_vector, F_matrix)
                backward_time = tm.time()
                self._trajectory, C_matrix, c_vector, F_matrix, _, isStop = self.forward_pass(
                    self._trajectory, K_matrix, k_vector)
                forward_time = tm.time()
                # do not care the value of log barrier
                obj = self._real_obj_fun.eval_obj_fun(self._trajectory)
                logger.info("[+ +] Total Iter.No.%3d   Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e" % (
                    total_iter_no,     i,  backward_time-iter_start_time, forward_time-backward_time, obj))
                logger.save_to_json(trajectory=self._trajectory.tolist())
                if isStop and self.kwargs['is_check_stop']:
                    self.set_obj_fun_value(np.inf)
                    logger.info(
                        "[+ +] Complete One Inner Loop! The log barrier parameter t is %.5f" % (j) + " in this iteration!")
                    break
        end_time = tm.time()
        logger.info("[+ +] Completed! All Time:%.5e" % (end_time-start_time))

    @property
    def obj_fun(self) -> iLQRObjectiveFunction:
        return self._obj_fun

    @property
    def dynamic_model(self) -> iLQRDynamicModel:
        return self._dynamic_model

    def staticmethod1(cls, a, b, c):
        print(aaaa)
    