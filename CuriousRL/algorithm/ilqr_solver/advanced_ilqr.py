#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import os
import torch
from .basic_ilqr import iLQRWrapper
from loguru import logger
from scipy.ndimage import gaussian_filter1d


class LogBarrieriLQR(iLQRWrapper):
    """This is an LogBarrieriLQR class
    """
    def __init__(self, dynamic_model, obj_fun):
        """ Initialization

            Parameter
            -----------
            dynamic_model : DynamicModelWrapper
                The dynamic model of the system
            obj_fun : ObjectiveFunctionWrapper
                The objective function of the iLQR
        """
        super().__init__(dynamic_model, obj_fun)

    def clear_obj_fun_value_last(self):
        self.obj_fun_value_last = np.inf

    def solve(self, example_name, max_iter = 100, is_check_stop = True):
        """ Solve the constraint problem with log barrier iLQR

            Parameter
            -----------
            example_name : string
                Name of the example
            max_iter : int
                The max iteration of iLQR
            is_check_stop : boolean
                Whether check the stopping criterion, if False, then max_iter number of iterations are performed
        """
        logger.debug("[+ +] Initial Obj.Val.: %.5e"%(self.get_obj_fun_value()))
        self.clear_obj_fun_value_last()
        self.backward_pass()
        self.forward_pass()
        start_time = tm.time()
        for j in [0.5, 1., 2., 5., 10., 20., 50., 100.]:
            self.obj_fun.update_t(j)
            for i in range(max_iter):
                iter_start_time = tm.time()
                self.backward_pass()
                backward_time = tm.time()
                obj, isStop = self.forward_pass(line_search = "feasibility")
                forward_time = tm.time()
                logger.debug("[+ +] Iter.No.%3d   BWTime:%.3e   FWTime:%.3e   Obj.Val.:%.5e"%(
                            i,  backward_time-iter_start_time,forward_time-backward_time,obj))
                result_path = os.path.join("logs", example_name, str(j) + "_" + str(i) + ".mat")
                io.savemat(result_path,{"trajectory": self.get_traj()})
                if isStop and is_check_stop:
                    self.clear_obj_fun_value_last()
                    logger.debug("[+ +] Complete One Inner Loop! The log barrier parameter t is %.5f"%(j) + " in this iteration!")
                    logger.debug("[+ +] Iteration No.\t Backward Time \t Forward Time \t Objective Value")
                    break
        end_time = tm.time()
        logger.debug("[+ +] Completed! All Time:%.5e"%(end_time-start_time))

class NNiLQR(iLQRWrapper):
    """This is an Neural Network iLQR class
    """
    def __init__(self, dynamic_model, obj_fun):
        """ Initialization

            Parameter
            -----------
            dynamic_model : DynamicModelWrapper
                The dynamic model of the system
            obj_fun : ObjectiveFunctionWrapper
                The objective function of the iLQR
        """
        super().__init__(dynamic_model, obj_fun)

    def solve(  self, 
                example_name, 
                nn_dynamic_model, 
                dataset_train, 
                re_train_stopping_criterion = 1e-5, 
                max_iter = 100, 
                max_line_search = 50,
                decay_rate = 1, 
                decay_rate_max_iters = 300,
                gaussian_filter_sigma = 10,
                gaussian_noise_sigma = 1):
        """ Solve the problem with data-driven iLQR

            Parameter
            -----------
            example_name : string
                Name of the example
            nn_dynamic_model : NeuralDynamicModelWrapper
                The neural network dynamic model
            dataset_train : DynamicModelDataSetWrapper
                Data set for training
            max_iter : int
                The max number of iterations of iLQR
            re_train_stopping_criterion : double
                Stopping criterion for re-training
            decay_rate : double
                Re_train_stopping_criterion = re_train_stopping_criterion * decay_rate
            decay_rate_max_iters : int
                The max iterations the decay_rate existing
            gaussian_filter_sigma : int
                Sigma parameter for the gaussian filter.
                The gaussian filter is for the F matrix
            gaussian_noise_sigma : tuple or int
                The gaussian noise injected into the system input when the trajectory is converged
        """
        logger.debug("[+ +] Initial Obj.Val.: %.5e"%(self.get_obj_fun_value()))
        trajectory = self.get_traj()
        new_data = []
        result_obj_val = np.zeros(max_iter)
        result_iter_time = np.zeros(max_iter)
        for i in range(int(max_iter)):
            if i == 1:  # skip the compiling time 
                start_time = tm.time()
            iter_start_time = tm.time()
            F_matrix = nn_dynamic_model.eval_grad_dynamic_model(trajectory)
            F_matrix = gaussian_filter1d(F_matrix, sigma = gaussian_filter_sigma, axis=0)
            self.update_F_matrix(F_matrix)
            self.backward_pass()
            obj_val, isStop = self.forward_pass(max_line_search=max_line_search)
            if i < decay_rate_max_iters:
                re_train_stopping_criterion = re_train_stopping_criterion * decay_rate
            iter_end_time = tm.time()
            iter_time = iter_end_time-iter_start_time
            logger.debug("[+ +] Iter.No.:%3d  Iter.Time:%.3e   Obj.Val.:%.5e"%(
                                    i,        iter_time,       obj_val,   ))
            result_obj_val[i] = obj_val
            result_iter_time[i] = iter_time
            trajectory = self.get_traj()
            if isStop: 
                if len(new_data) != 0: # Ensure the optimal trajectroy being in the dataset
                    trajectory_noisy = trajectory
                else:
                    trajectory_noisy = self.dynamic_model.eval_traj(input_traj = (trajectory[:,self.dynamic_model.n:]+np.random.normal(0, gaussian_noise_sigma, [self.dynamic_model.T,self.dynamic_model.m,1])))
                new_data += [trajectory_noisy]
                dataset_train.update_dataset(new_data[-int(dataset_train.Trial_No/5):]) # at most update 20% dataset
                result_path = os.path.join("logs", example_name, str(i) +".mat")
                io.savemat(result_path,{"optimal_trajectory": self.get_traj(), "trajectroy_noisy": trajectory_noisy})
                nn_dynamic_model.retrain(dataset_train, max_epoch = 100000, stopping_criterion = re_train_stopping_criterion)
                new_data = []
            else: 
                new_data += [trajectory]
        end_time = tm.time()
        io.savemat(os.path.join("logs", example_name,  "_result.mat"),{"obj_val": result_obj_val, "iter_time": result_iter_time})
        logger.debug("[+ +] Completed! All Time:%.5e"%(end_time-start_time))

# %%
