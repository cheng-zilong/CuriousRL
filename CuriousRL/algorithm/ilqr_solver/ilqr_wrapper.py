from __future__ import annotations
import numpy as np
import abc
from numba import njit
from CuriousRL.algorithm.algo_wrapper import AlgoWrapper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .ilqr_obj_fun import iLQRObjectiveFunction
    from .ilqr_dynamic_model import iLQRDynamicModel

class iLQRWrapper(AlgoWrapper):
    """This is a wrapper class for all the iLQR algorithms. 
    This class is an abstract class but provides several basic methods for the iLQR iteration. 
    This class cannot create algorithm instance. 
    To create instance, the abstract methods ``solve``, ``get_obj_fun``, ``get_dynamic_model`` are required to be realized.
    Initialize the fundamental parameters. All child classes must call ``super().__init__(...)`` to initalize these parameters 
    because all these parameters are required in the basic iLQR algorithm. 

    :param max_iter: Maximum number of the iLQR iterations.
    :type max_iter: int
    :param is_check_stop: Decide whether the stopping criterion is checked.
        If is_check_stop = False, then the maximum number of the iLQR iterations will be reached.
    :type is_check_stop: bool
    :param stopping_criterion: When stopping criterion is satisfied, then the current iLQR algorithm will stop.
        If is_check_stop = False, then the stopping_criterion will be suppressed.
    :type stopping_criterion: double
    :param max_line_search:  Maximum number of line search iterations.
    :type max_line_search: int
    :param gamma: Gamma is the parameter for the line search, that is alpha=gamma*alpha if line search 
        requirements are not satisfied, where alpha is the step size of the current iteration.
    :type gamma: double
    :param line_search_method: The method for line search. The provided line search methods are listed as bellow.
        ``"vanilla"``: The current objective function value is smaller than the last
        ``"feasibility"``: The current objective function value is smaller than the last and the trajectory is feasible
        ``"none"``: No line search
    :type line_search_method: str
    :param stopping_method: The method for stopping the iteration. The provided stopping methods are listed as bellow.
        ``"vanilla"``: The difference between the last objective function value and the current objective function value is smaller than the stopping_criterion
        ``"relative"``: The difference between the last objective function value and the current objective function value relative 
        to the last objective function value is smaller than the stopping_criterion
    :type stopping_method: str
    """

    def __init__(self, max_iter, is_check_stop, stopping_criterion, max_line_search, gamma, line_search_method, stopping_method):
        super().__init__(max_iter=max_iter,
                         is_check_stop=is_check_stop,
                         stopping_criterion=stopping_criterion,
                         max_line_search=max_line_search,
                         gamma=gamma,
                         line_search_method=line_search_method,
                         stopping_method=stopping_method)
        self.max_iter = max_iter
        self.is_check_stop = is_check_stop
        self.stopping_criterion = stopping_criterion
        self.max_line_search = max_line_search
        self.gamma = gamma
        self.line_search_method = line_search_method
        self.stopping_method = stopping_method

    @abc.abstractmethod
    def get_obj_fun(self) -> iLQRObjectiveFunction:
        """ Return the objective function for the iLQR algorithm.

        :return: objective function 
        :rtype: iLQRObjectiveFunction
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dynamic_model(self) -> iLQRDynamicModel:
        """ Return the dynamic model function for the iLQR algorithm.

        :return: dynamic model function
        :rtype: iLQRDynamicModel
        """
        raise NotImplementedError

    def _vanilla_line_search(self):
        """The line search method to ensure the value of the objective function is reduced monotonically.

        :return: The current_iteration_trajectory after line search. 
        :rtype: array(T, m+n, 1)
        :return: The value of the objective function after the line search
        :rtype: double
        """
        # alpha: Step size
        alpha = 1.
        trajectory_current = np.zeros((self.T, self.n+self.m, 1))
        # Line Search if the z value is greater than zero
        for _ in range(self.max_line_search):
            trajectory_current = self.get_dynamic_model().update_traj(
                self.trajectory, self.K_matrix, self.k_vector, alpha)
            obj_fun_value_current = self.get_obj_fun().eval_obj_fun(trajectory_current)
            obj_fun_value_delta = obj_fun_value_current-self.obj_fun_value_last
            alpha = alpha * self.gamma
            if obj_fun_value_delta < 0:
                return trajectory_current, obj_fun_value_current
        return self.trajectory, self.obj_fun_value_last

    def _feasibility_line_search(self):
        """To ensure the value of the objective function is reduced monotonically, and ensure the trajectory for the next iteration is feasible.

        :return: The current_iteration_trajectory after line search. 
        :rtype: array(T, m+n, 1)
        :return: The value of the objective function after the line search
        :rtype: double
        """
        # alpha: Step size
        alpha = 1.
        trajectory_current = np.zeros((self.T, self.n+self.m, 1))
        # Line Search if the z value is greater than zero
        for _ in range(self.max_line_search):
            trajectory_current = self.get_dynamic_model().update_traj(
                self.trajectory, self.K_matrix, self.k_vector, alpha)
            obj_fun_value_current = self.get_obj_fun().eval_obj_fun(trajectory_current)
            obj_fun_value_delta = obj_fun_value_current-self.obj_fun_value_last
            alpha = alpha * self.gamma
            if obj_fun_value_delta < 0 and (not np.isnan(obj_fun_value_delta)):
                return trajectory_current, obj_fun_value_current
        return self.trajectory, self.obj_fun_value_last

    def _none_line_search(self):
        """ Do not use any line search method

        :return: The current_iteration_trajectory after line search. 
        :rtype: array(T, m+n, 1)
        :return: The value of the objective function after the line search
        :rtype: double
        """
        trajectory_current = self.get_dynamic_model().update_traj(
            self.trajectory, self.K_matrix, self.k_vector, 1)
        obj_fun_value_current = self.get_obj_fun().eval_obj_fun(trajectory_current)
        return trajectory_current, obj_fun_value_current

    def _vanilla_stopping_criterion(self, obj_fun_value_current):
        """Check the amount of change of the objective function. If the amount of change 
        is less than the specific value, the stopping criterion is satisfied.

        :param obj_fun_value_current: Current objective function value
        :type obj_fun_value_current: double
        :return: Whether the stopping criterion is reached. True: the stopping criterion is satisfied
        :rtype: bool
        """
        obj_fun_value_delta = obj_fun_value_current - self.obj_fun_value_last
        if (abs(obj_fun_value_delta) < self.stopping_criterion):
            return True
        return False

    def _relative_stopping_criterion(self, obj_fun_value_current):
        """ Check the amount of change of the objective function relative to the current objective function value. 
        If the amount of change is less than the specific value, the stopping criterion is satisfied.
        
        :param obj_fun_value_current: Current objective function value
        :type obj_fun_value_current: double
        :return: Whether the stopping criterion is reached. True: the stopping criterion is satisfied
        :rtype: bool
        """
        obj_fun_value_delta = obj_fun_value_current - self.obj_fun_value_last
        if (abs(obj_fun_value_delta/self.obj_fun_value_last) < self.stopping_criterion):
            return True
        return False

    def forward_pass(self):
        """Forward pass in the iLQR algorithm.

        :return: The value of the objective function after the line search
        :rtype: double
        :return: Whether the stopping criterion is reached. True: the stopping criterion is satisfied
        :rtype: bool
        """
        # Do line search
        if self.line_search_method == "vanilla":
            self.trajectory, obj_fun_value_current = self._vanilla_line_search()
        elif self.line_search_method == "feasibility":
            self.trajectory, obj_fun_value_current = self._feasibility_line_search()
        elif self.line_search_method == "none":
            self.trajectory, obj_fun_value_current = self._none_line_search()
        # Check the stopping criterion
        if self.stopping_method == "vanilla":
            is_stop = self._vanilla_stopping_criterion(obj_fun_value_current)
        elif self.stopping_method == "relative":
            is_stop = self._relative_stopping_criterion(obj_fun_value_current)
        # Do forward pass
        self.C_matrix = self.get_obj_fun().eval_hessian_obj_fun(self.trajectory)
        self.c_vector = self.get_obj_fun().eval_grad_obj_fun(self.trajectory)
        self.F_matrix = self.get_dynamic_model().eval_grad_dynamic_model(self.trajectory)
        # Finally update the objective_function_value_last
        self.obj_fun_value_last = obj_fun_value_current
        return obj_fun_value_current, is_stop

    def backward_pass(self):
        """Backward pass in the iLQR algorithm.

        :return: feedback matrix K
        :rtype: array(T, m, n)
        :return: feedforward vector k
        :rtype: array(T, m, 1)
        """
        self.K_matrix, self.k_vector = self._backward_pass_static(
            self.m, self.n, self.T, self.C_matrix, self.c_vector, self.F_matrix)
        return self.K_matrix, self.k_vector

    @staticmethod
    @njit
    def _backward_pass_static(m, n, T, C_matrix, c_vector, F_matrix):
        V_matrix = np.zeros((n, n))
        v_vector = np.zeros((n, 1))
        K_matrix_list = np.zeros((T, m, n))
        k_vector_list = np.zeros((T, m, 1))
        for i in range(T-1, -1, -1):
            Q_matrix = C_matrix[i] + F_matrix[i].T@V_matrix@F_matrix[i]
            q_vector = c_vector[i] + F_matrix[i].T@v_vector
            Q_uu = Q_matrix[n:n+m, n:n+m].copy()
            Q_ux = Q_matrix[n:n+m, 0:n].copy()
            q_u = q_vector[n:n+m].copy()

            K_matrix_list[i] = -np.linalg.solve(Q_uu, Q_ux)
            k_vector_list[i] = -np.linalg.solve(Q_uu, q_u)
            V_matrix = Q_matrix[0:n, 0:n] +\
                Q_ux.T@K_matrix_list[i] +\
                K_matrix_list[i].T@Q_ux +\
                K_matrix_list[i].T@Q_uu@K_matrix_list[i]
            v_vector = q_vector[0:n] +\
                Q_ux.T@k_vector_list[i] +\
                K_matrix_list[i].T@q_u +\
                K_matrix_list[i].T@Q_uu@k_vector_list[i]
        return K_matrix_list, k_vector_list

    def get_obj_fun_value(self):
        """Return the current objective function value.

        :return: Current objective function value
        :rtype: double
        """
        return self.obj_fun_value_last

    def reset_obj_fun_value(self):
        """ Reset the value of the objective function to be the initial objective function value.
        """
        self.obj_fun_value_last = self.init_obj

    def set_obj_fun_value(self, obj_fun_value):
        """Set the value of the objective function.

        :param obj_fun_value: New objective function value
        :type obj_fun_value: double
        """
        self.obj_fun_value_last = obj_fun_value

    def get_traj(self):
        """Return the current trajectory.

        :return: Current trajectory
        :rtype: array(T_int, m+n, 1)
        """
        return self.trajectory.copy()

    def set_F_matrix(self, F_matrix):
        """Set the feedback matrix F in iLQR method

        :param F_matrix: The new feedback matrix F
        :type F_matrix: array(T_int, n, m+n)
        """
        self.F_matrix = F_matrix

    def set_init_state(self, new_state):
        """Set the init state of the dynamic system

        :param new_state: The new state
        :type new_state: array(n, 1)
        """
        self.get_dynamic_model().init_state = new_state

    def set_obj_add_param(self, new_add_param):
        """Set the values of the additional parameters in the objective function

        :param new_add_param: The new values to the additioanl variables
        :type new_add_param: array(T, p)
        """
        self.get_obj_fun().add_param = new_add_param

    def get_obj_add_param(self):
        """Get the values of the additional parameters in the objective function

        :return: additional parameters in the objective function
        :rtype: array(T, p)
        """
        return self.get_obj_fun().add_param