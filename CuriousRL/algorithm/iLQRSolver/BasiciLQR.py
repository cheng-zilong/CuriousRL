#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import cvxpy as cp
from numba import njit, jitclass, jit
import numba
from loguru import logger
import os 

class iLQRWrapper(object):
    """This is a wrapper class for the iLQR iteraton
    """
    def __init__(self, dynamic_model, obj_fun):
        """ Initialization the iLQR solver class

            Parameter
            -----------
            dynamic_model : DynamicModelWrapper
                The dynamic model of the system
            obj_fun : ObjectiveFunctionWrapper
                The objective function of the iLQR
        """
        # Initialize the functions
        self.dynamic_model = dynamic_model
        self.obj_fun = obj_fun
        # Parameters for the model
        self.n = dynamic_model.n
        self.m = dynamic_model.m
        self.T = dynamic_model.T
        # Initialize the trajectory, F_matrix, objective_function_value_last, C_matrix and c_vector
        self.trajectory = self.dynamic_model.eval_traj()
        self.F_matrix = self.dynamic_model.eval_grad_dynamic_model(self.trajectory)
        self.init_obj = self.obj_fun.eval_obj_fun(self.trajectory)
        self.obj_fun_value_last = self.init_obj
        self.c_vector = self.obj_fun.eval_grad_obj_fun(self.trajectory)
        self.C_matrix = self.obj_fun.eval_hessian_obj_fun(self.trajectory)

    def get_traj(self):
        """ Return the current trajectory

            Return
            -----------
            trajectory : array(T_int, m+n, 1)
                Current trajectory
        """
        return self.trajectory.copy()
        
    def update_F_matrix(self, F_matrix):
        """ Update F matrix in iLQR method

            Parameter
            -----------
            F_matrix : array(T_int, n, m+n)
                The new F matrix
        """
        self.F_matrix = F_matrix

    def _vanilla_line_search(self,  gamma, maximum_line_search):
        """To ensure the value of the objective function is reduced monotonically

            Parameters
            ----------
            gamma : double 
                Gamma is the parameter for the line search : alpha=gamma*alpha

            Return
            ----------
            current_trajectory : array(T, m+n, 1)
                The current_iteration_trajectory after line search.
            current_objective_function_value : double
                The value of the objective function after the line search
        """
        # alpha: Step size
        alpha = 1.
        trajectory_current = np.zeros((self.T, self.n+self.m, 1))
        for _ in range(maximum_line_search): # Line Search if the z value is greater than zero
            trajectory_current = self.dynamic_model.update_traj(self.trajectory, self.K_matrix, self.k_vector, alpha)
            obj_fun_value_current = self.obj_fun.eval_obj_fun(trajectory_current)
            obj_fun_value_delta = obj_fun_value_current-self.obj_fun_value_last
            alpha = alpha * gamma
            if obj_fun_value_delta<0:
                return trajectory_current, obj_fun_value_current
        return self.trajectory, self.obj_fun_value_last
        
    def _feasibility_line_search(self, gamma, maximum_line_search):
        """To ensure the value of the objective function is reduced monotonically, and ensure the trajectory for the next iteration is feasible.

            Parameters
            ----------
            gamma : double 
                Gamma is the parameter for the line search : alpha=gamma*alpha

            Return
            ----------
            current_trajectory : float64[T,m+n,1]
                The current_iteration_trajectory after line search.
            current_objective_function_value : float64
                The value of the objective function after the line search
        """
        # alpha: Step size
        alpha = 1.
        trajectory_current = np.zeros((self.T, self.n+self.m, 1))
        for _ in range(maximum_line_search): # Line Search if the z value is greater than zero
            trajectory_current = self.dynamic_model.update_traj(self.trajectory, self.K_matrix, self.k_vector, alpha)
            obj_fun_value_current = self.obj_fun.eval_obj_fun(trajectory_current)
            obj_fun_value_delta = obj_fun_value_current-self.obj_fun_value_last
            alpha = alpha * gamma
            if obj_fun_value_delta<0 and (not np.isnan(obj_fun_value_delta)):
                return trajectory_current, obj_fun_value_current
        return self.trajectory, self.obj_fun_value_last
    
    def _none_line_search(self):
        """ Do not use any line search method

            Return
            ----------
            current_trajectory : float64[T,m+n,1]
                The current_iteration_trajectory after line search.
            current_objective_function_value : float64
                The value of the objective function after the line search
        """
        trajectory_current = self.dynamic_model.update_traj(self.trajectory, self.K_matrix, self.k_vector, 1)
        obj_fun_value_current = self.obj_fun.eval_obj_fun(trajectory_current)
        return trajectory_current, obj_fun_value_current

    def _vanilla_stopping_criterion(self, obj_fun_value_current, stopping_criterion):
        """Check the amount of change of the objective function. If the amount of change is less than the specific value, the stopping criterion is satisfied.

            Parameters
            ----------
            delta_objective_function_value : double
                The delta_objective_function_value in the current iteration.

            stopping_criterion : double 
                The number of input variables

            Return
            ----------
            isStop: Boolean
                Whether the stopping criterion is reached. True: the stopping criterion is satisfied
        """
        obj_fun_value_delta = obj_fun_value_current - self.obj_fun_value_last
        if (abs(obj_fun_value_delta) < stopping_criterion):
            return True
        return False
        
    def forward_pass(self, gamma = 0.5, max_line_search = 50, line_search = "vanilla", stopping_method = "vanilla", stopping_criterion = 1e-6):
        """Forward_pass in the iLQR algorithm with simple line search
        
            Parameters
            ----------
            gamma : double 
                Gamma is the parameter for the line search: alpha=gamma*alpha
            max_line_search : int
                Maximum iterations of line search
            line_search : string 
                Line search method ("vanilla", "feasibility", None)
            stopping_method : string
                Stopping method
            stopping_criterion : double 
                Stopping Criterion

            Return
            ----------
            current_obj_fun_value: double
                The value of the objective function after the line search
            is_stop: Boolean
                Whether the stopping criterion is reached. True: the stopping criterion is satisfied
            C_matrix : array(T, n + m, n + m)

            c_vector : array(T, n + m, n)

            F_matrix : array(T, n, n  + m)
        """
        # Do line search
        if line_search == "vanilla":
            self.trajectory, obj_fun_value_current = self._vanilla_line_search(gamma, max_line_search)
        elif line_search == "feasibility":
            self.trajectory, obj_fun_value_current = self._feasibility_line_search(gamma, max_line_search)
        elif line_search == None:
            self.trajectory, obj_fun_value_current = self._none_line_search()
        # Check the stopping criterion
        if stopping_method == "vanilla":
            is_stop = self._vanilla_stopping_criterion(obj_fun_value_current, stopping_criterion)
        # Do forward pass
        self.C_matrix = self.obj_fun.eval_hessian_obj_fun(self.trajectory)
        self.c_vector = self.obj_fun.eval_grad_obj_fun(self.trajectory)
        self.F_matrix = self.dynamic_model.eval_grad_dynamic_model(self.trajectory)
        # Finally update the objective_function_value_last
        self.obj_fun_value_last = obj_fun_value_current
        return obj_fun_value_current, is_stop

    def backward_pass(self):
        """Backward_pass in the iLQR algorithm

            Return
            ------------
            K_matrix : array(T, m, n)
                K matrix in iLQR
            k_vector : array(T, m, 1)
                k vector in iLQR
        """
        self.K_matrix, self.k_vector = self.backward_pass_static(self.m, self.n, self.T, self.C_matrix, self.c_vector, self.F_matrix)
        return self.K_matrix, self.k_vector

    @staticmethod
    @njit
    def backward_pass_static(m, n, T, C_matrix, c_vector, F_matrix):
        V_matrix = np.zeros((n,n))
        v_vector = np.zeros((n,1))
        K_matrix_list = np.zeros((T, m, n))
        k_vector_list = np.zeros((T, m, 1))
        for i in range(T-1,-1,-1):
            Q_matrix = C_matrix[i] + F_matrix[i].T@V_matrix@F_matrix[i]
            q_vector = c_vector[i] + F_matrix[i].T@v_vector
            Q_uu = Q_matrix[n:n+m,n:n+m].copy()
            Q_ux = Q_matrix[n:n+m,0:n].copy()
            q_u = q_vector[n:n+m].copy()

            K_matrix_list[i] = -np.linalg.solve(Q_uu,Q_ux)
            k_vector_list[i] = -np.linalg.solve(Q_uu,q_u)
            V_matrix = Q_matrix[0:n,0:n]+\
                            Q_ux.T@K_matrix_list[i]+\
                            K_matrix_list[i].T@Q_ux+\
                            K_matrix_list[i].T@Q_uu@K_matrix_list[i]
            v_vector = q_vector[0:n]+\
                            Q_ux.T@k_vector_list[i] +\
                            K_matrix_list[i].T@q_u +\
                            K_matrix_list[i].T@Q_uu@k_vector_list[i]
        return K_matrix_list, k_vector_list

    def get_obj_fun_value(self):
        return self.obj_fun_value_last

    def clear_obj_fun_value_last(self):
        self.obj_fun_value_last = self.init_obj
    
    def solve(self, example_name, max_iter = 100, is_check_stop = True, maximum_line_search = 10):
        """ Solve the problem with classical iLQR

            Parameter
            -----------
            example_name : string
                Name of the example
            max_iter : int
                The max number of iterations of iLQR
            is_check_stop : boolean
                Whether check the stopping criterion, if False, then max_iter number of iterations are performed
        """
        logger.debug("[+ +] Initial Obj.Val.: %.5e"%(self.get_obj_fun_value()))
        start_time = tm.time()
        for i in range(max_iter):
            if i == 1:  # skip the compiling time 
                start_time = tm.time()
            iter_start_time = tm.time()
            self.backward_pass()
            backward_time = tm.time()
            obj, isStop = self.forward_pass(max_line_search=maximum_line_search)
            forward_time = tm.time()
            logger.debug("[+ +] Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e"%(
                         i,  backward_time-iter_start_time,forward_time-backward_time,obj))
            result_path = os.path.join("logs", example_name, str(i) +".mat")
            io.savemat(result_path,{"trajectory": self.get_traj()})
            if isStop and is_check_stop:
                break
        end_time = tm.time()
        logger.debug("[+ +] Completed! All Time:%.5e"%(end_time-start_time))
#%%iLQR