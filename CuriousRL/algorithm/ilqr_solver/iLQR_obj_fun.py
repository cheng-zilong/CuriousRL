import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import cvxpy as cp
from numba import njit, jitclass, jit
import numba

class iLQRObjectiveFunction(object):
    """This is a wrapper class for the objective function"""
    def __init__(self, obj_fun, x_u_var, add_param_var = None, add_param = None):
        """ Initialization

            Parameters
            -----------------
            obj_fun : function(x_u, (additional_variables))
                The function of the objetive function
                The methods in this class will only be related to the first argument, 
                The second optional argument is for the purpose of new methods design
            x_u_var : tuple with sympy.symbols 
                State and input variables in the objective function
            add_param_var : tuple with sympy.symbols 
                Introduce the additional variables that are not derived
            add_param : array(T, -1)
                Give the values to the additioanl variables
        """
        if add_param_var is None:
            add_param_var = sp.symbols("no_use")
        self.obj_fun_lamdify = njit(sp.lambdify([x_u_var,add_param_var], obj_fun, "numpy"))
        gradient_objective_function_array = sp.derive_by_array(obj_fun, x_u_var)
        self.grad_obj_fun_lamdify = njit(sp.lambdify([x_u_var, add_param_var], gradient_objective_function_array,"numpy"))       
        hessian_objective_function_array = sp.derive_by_array(gradient_objective_function_array, x_u_var)
        # A stupid method to ensure each element in the hessian matrix is in the type of float64
        self.hessian_obj_fun_lamdify = njit(sp.lambdify([x_u_var, add_param_var], np.asarray(hessian_objective_function_array)+1e-100*np.eye(hessian_objective_function_array.shape[0]),"numpy"))
        self.add_param = add_param

    def update_add_param(self, new_add_param):
        """ Update the additional parameters in the objective function

            Parameters
            -----------------
            add_param : array(T, -1)
                Values to the additioanl variables
        """
        self.add_param = new_add_param

    def get_add_param(self):
        return self.add_param

    def eval_obj_fun(self, trajectory):
        """Return the objective function value

            Parameters
            -----------------
            trajectory : array(T, m+n, 1) 

            Return
            ---------------
            obj : scalar
                The objective function value
        """
        return self._eval_obj_fun_static(self.obj_fun_lamdify, trajectory, self.add_param)

    def eval_grad_obj_fun(self, trajectory):
        """ Return the objective function value

            Parameters
            -----------------
            trajectory : array(T, m+n, 1) 

            Return
            ---------------
            grad : array[T, m+n,1] 
                The objective function jacobian
        """
        return self._eval_grad_obj_fun_static(self.grad_obj_fun_lamdify, trajectory, self.add_param)

    def eval_hessian_obj_fun(self, trajectory):
        """ Return the objective function value

            Parameters
            -----------------
            trajectory : array(T, m+n, 1) 

            Return
            ---------------
            grad : array[T, m+n, m+n] 
                The objective function hessian
        """
        return self._eval_hessian_obj_fun_static(self.hessian_obj_fun_lamdify, trajectory, self.add_param)

    @staticmethod
    @njit
    def _eval_obj_fun_static(obj_fun_lamdify, trajectory, add_param):
        T = int(trajectory.shape[0])
        if add_param is None:
            add_param = np.zeros((T,1))
        obj_value = 0.
        for tau in range(T):
            obj_value = obj_value + np.asarray(obj_fun_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64)
        return obj_value
    
    @staticmethod
    @njit
    def _eval_grad_obj_fun_static(grad_obj_fun_lamdify, trajectory, add_param):
        T = int(trajectory.shape[0])
        m_n = int(trajectory.shape[1])
        if add_param is None:
            add_param = np.zeros((T,1))
        grad_all_tau = np.zeros((T, m_n, 1))
        for tau in range(T):
            grad_all_tau[tau] = np.asarray(grad_obj_fun_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64).reshape(-1,1)
        return grad_all_tau
    
    @staticmethod
    @njit
    def _eval_hessian_obj_fun_static(hessian_obj_fun_lamdify, trajectory, add_param):
        T = int(trajectory.shape[0])
        m_n = int(trajectory.shape[1])
        if add_param is None:
            add_param = np.zeros((T,1))
        hessian_all_tau = np.zeros((T, m_n, m_n))
        for tau in range(T):
            hessian_all_tau[tau] = np.asarray(hessian_obj_fun_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64)
        return hessian_all_tau
