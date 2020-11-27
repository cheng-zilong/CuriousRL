import numpy as np
import sympy as sp
from numba import njit

class iLQRDynamicModel(object):
    """ This is a wrapper class for the dynamic model
    """
    def __init__(self, dynamic_function, x_u_var, constr, init_state, init_input, add_param_var = None, add_param = None):
        """ Initialization
            
            Parameters
            ---------------
            dynamic_function : sympy.array with symbols
                The model dynamic function defined by sympy symbolic array
            x_u_var : tuple with sympy.symbols 
                State and input variables in the model
            init_state : array(n, 1)
                The initial state vector of the system
            init_input : array(T, m, 1) 
                The initial input vector
            add_param_var : tuple with sympy.symbols 
                Introduce the additional variables that are not derived
            add_param : array(T, -1)
                Give the values to the additioanl variables
        """
        self.init_state = init_state
        self.init_input = init_input
        self.n = int(init_state.shape[0])
        self.m = int(len(x_u_var) - self.n)
        self.T = int(init_input.shape[0])
        if add_param_var is None:
            add_param_var = sp.symbols("no_use")
        self.dynamic_function_lamdify = njit(sp.lambdify([x_u_var, add_param_var], dynamic_function, "math"))
        grad_dynamic_function = sp.transpose(sp.derive_by_array(dynamic_function, x_u_var))
        self.grad_dynamic_function_lamdify = njit(sp.lambdify([x_u_var, add_param_var], grad_dynamic_function, "math"))
        self.add_param = add_param
        self.constr = constr

    def eval_traj(self, init_state = None, input_traj = None):
        """ Evaluate the system trajectory by given initial states and input vector
            Parameters
            -----------------
            init_state : array(n, 1)
                The initial state used to evaluate trajectory
            input_traj : array(T, n, 1)
                The input trajectory used to evaluate trajectory

            Return
            ---------------
            trajectory : array(T, m+n, 1)
                The whole trajectory
        """
        if init_state is None:
            init_state = self.init_state
        if input_traj is None:
            input_traj = self.init_input
        return self._eval_traj_static(self.dynamic_function_lamdify, init_state, input_traj, self.add_param, self.m, self.n, self.constr)

    def update_traj(self, old_traj, K_matrix_all, k_vector_all, alpha): 
        """ Update the trajectory by using iLQR
            Parameters
            -----------------
            old_traj : array(T, m+n, 1)
                The trajectory in the last iteration
            K_matrix_all : array(T, m, n)\\
            k_vector_all : array(T, m, 1)\\
            alpha : double
                Step size in this iteration

            Return
            ---------------
            new_trajectory : array(T, m+n, 1) 
                The updated trajectory
        """
        return self._update_traj_static(self.dynamic_function_lamdify, self.m, self.n, old_traj, K_matrix_all, k_vector_all, alpha, self.add_param, self.constr)

    def eval_grad_dynamic_model(self, trajectory):
        """ Return the matrix of the gradient of the dynamic_model
            Parameters
            -----------------
            trajectory : array(T, m+n, 1)
                System trajectory

            Return
            ---------------
            grad : array(T, m, n)
                The gradient of the dynamic_model
        """
        return self._eval_grad_dynamic_model_static(self.grad_dynamic_function_lamdify, trajectory, self.add_param)

    @staticmethod
    @njit
    def _eval_traj_static(dynamic_model_lamdify, init_state, input_traj, add_param, m, n, constr):
        T = int(input_traj.shape[0])
        if add_param == None:
            add_param = np.zeros((T,1))
        trajectory = np.zeros((T, m+n, 1))
        trajectory[0] = np.vstack((init_state, input_traj[0]))
        for tau in range(T-1):
            trajectory[tau+1, :n, 0] = np.asarray(dynamic_model_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64)
            trajectory[tau+1, n:] = input_traj[tau+1]
            for i, c in enumerate(constr):
                trajectory[tau, i, 0] = min(max(c[0], trajectory[tau, i, 0]), c[1]) 
        return trajectory

    @staticmethod
    @njit
    def _update_traj_static(dynamic_model_lamdify, m, n, old_traj, K_matrix_all, k_vector_all, alpha, add_param, constr):
        T = int(K_matrix_all.shape[0])
        if add_param == None:
            add_param = np.zeros((T,1))
        new_trajectory = np.zeros((T, m+n, 1))
        new_trajectory[0] = old_traj[0] # initial states are the same
        for tau in range(T-1):
            # The amount of change of state x
            delta_x = new_trajectory[tau, 0:n] - old_traj[tau, 0:n]
            # The amount of change of input u
            delta_u = K_matrix_all[tau]@delta_x+alpha*k_vector_all[tau]
            # The real input of next iteration
            input_u = old_traj[tau, n:n+m] + delta_u
            new_trajectory[tau,n:] = input_u
            for i, c in enumerate(constr[n:]):
                new_trajectory[tau, n+i, 0] = min(max(c[0], new_trajectory[tau, n+i, 0]), c[1]) 
            new_trajectory[tau+1,:n] = np.asarray(dynamic_model_lamdify(new_trajectory[tau,:,0], add_param[tau]),dtype=np.float64).reshape(-1,1)
            for i, c in enumerate(constr[:n]):
                new_trajectory[tau+1, i, 0] = min(max(c[0], new_trajectory[tau+1, i, 0]), c[1]) 
            
            # dont care the input at the last time stamp, because it is always zero
        return new_trajectory

    @staticmethod
    @njit
    def _eval_grad_dynamic_model_static(grad_dynamic_model_lamdify, trajectory, add_param):
        T = int(trajectory.shape[0])
        if add_param == None:
            add_param = np.zeros((T,1))
        F_matrix_initial =  grad_dynamic_model_lamdify(trajectory[0,:,0], add_param[0])
        F_matrix_all = np.zeros((T, len(F_matrix_initial), len(F_matrix_initial[0])))
        F_matrix_all[0] = np.asarray(F_matrix_initial, dtype = np.float64)
        for tau in range(1, T):
            F_matrix_all[tau] = np.asarray(grad_dynamic_model_lamdify(trajectory[tau,:,0], add_param[tau]), dtype = np.float64)
        return F_matrix_all

