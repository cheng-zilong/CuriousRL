#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
from iLQRSolver import DynamicModel, ObjectiveFunction, BasiciLQR, AdvancediLQR, Logger
from loguru import logger
from datetime import datetime
import torch
from torch import nn, optim
from torch.autograd.functional import jacobian
from torch.utils.tensorboard import SummaryWriter
import sys
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - {message}")

class Residual(nn.Module):  
    """The Residual block of ResNet."""
    def __init__(self, input_channels, output_channels, is_shorcut = True):
        super().__init__()
        self.is_shorcut = is_shorcut
        self.bn1 = nn.BatchNorm1d(input_channels)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_channels, output_channels)
        
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.linear2 = nn.Linear(output_channels, output_channels)
    
        self.shorcut = nn.Linear(input_channels, output_channels)
        self.main_track = nn.Sequential(self.bn1, self.relu , self.linear1, self.bn2,  self.relu, self.linear2)
    def forward(self, X):
        if self.is_shorcut:
            Y = self.main_track(X) + self.shorcut(X)
        else:
            Y = self.main_track(X) + X
        return torch.nn.functional.relu(Y)
        
class SmallResidualNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        layer1_no=64
        layer2_no=32
        layer3_no=16
        layer4_no=8

        self.layer = nn.Sequential( Residual(in_dim, layer1_no),
                                    Residual(layer1_no, layer2_no),
                                    Residual(layer2_no, layer3_no),
                                    Residual(layer3_no, layer4_no),
                                    nn.Linear(layer4_no, out_dim))

    def forward(self, x):
        x = self.layer(x)
        return x


class SmallNetwork(nn.Module):
    """Here is a dummy network that can work well on the vehicle model
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        layer1_no=128
        layer2_no=64
        self.layer = nn.Sequential( 
                                    nn.Linear(in_dim, layer1_no), nn.BatchNorm1d(layer1_no), nn.ReLU(), 
                                    nn.Linear(layer1_no, layer2_no), nn.BatchNorm1d(layer2_no), nn.ReLU(), 
                                    nn.Linear(layer2_no, out_dim))
    def forward(self, x):
        x = self.layer(x)
        return x

class LargeNetwork(nn.Module):
    """Here is a dummy network that can work well on the vehicle model
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        layer1_no=800
        layer2_no=400
        self.layer = nn.Sequential( 
                                    nn.Linear(in_dim, layer1_no), nn.BatchNorm1d(layer1_no), nn.ReLU(), 
                                    nn.Linear(layer1_no, layer2_no), nn.BatchNorm1d(layer2_no), nn.ReLU(), 
                                    nn.Linear(layer2_no, out_dim))
    def forward(self, x):
        x = self.layer(x)
        return x

def cartpole1_vanilla(T = 300, max_iter = 10000, is_check_stop = True):
    file_name = "cartpole1_vanilla_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    #################################
    ######### Dynamic Model #########
    #################################
    cartpole, x_u, n, m = DynamicModel.cart_pole()
    init_state = np.asarray([np.pi,0,0,0],dtype=np.float64).reshape(-1,1)
    init_input = np.zeros((T,m,1))
    dynamic_model = DynamicModel.DynamicModelWrapper(cartpole, x_u, init_state, init_input, T)
    #################################
    ##### Objective Function ########
    #################################
    C_matrix_diag = sp.symbols("c:5")
    add_param_obj = np.zeros((T, 5), dtype = np.float64)
    for tau in range(T):
        if tau < T-1:
            add_param_obj[tau] = np.asarray((1, 0.1, 1, 1, 0.1))
        else: 
            add_param_obj[tau] = np.asarray((10000, 1000, 0, 0, 0))
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u)@np.diag(np.asarray(C_matrix_diag))@(x_u), x_u_var = x_u, add_param_var=C_matrix_diag, add_param=add_param_obj)
    #################################
    ######### iLQR Solver ###########
    #################################
    logger_id = Logger.loguru_start(        file_name = file_name, 
                                            T=T, 
                                            max_iter = max_iter, 
                                            is_check_stop = is_check_stop,
                                            init_state = init_state,
                                            obj_C = add_param_obj[0],
                                            obj_terminal_C = add_param_obj[-1])
    cartpole1_example = BasiciLQR.iLQRWrapper(dynamic_model, objective_function)
    cartpole1_example.solve(file_name, max_iter = max_iter, is_check_stop = is_check_stop)
    Logger.loguru_end(logger_id)

def cartpole1_dd_iLQR(  T = 300, 
                        trial_no = 100,
                        stopping_criterion = 1e-4,
                        max_iter=100,
                        max_line_search = 50,
                        decay_rate=0.99,
                        decay_rate_max_iters=300,
                        gaussian_filter_sigma = 10,
                        gaussian_noise_sigma = 1,
                        network = "small"):
    file_name = "cartpole1_dd_iLQR_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    #################################
    ######### Dynamic Model #########
    #################################
    cartpole, x_u, n, m = DynamicModel.cart_pole()
    init_state = np.asarray([np.pi, 0, 0, 0],dtype=np.float64).reshape(-1,1)
    init_input = np.zeros((T, m, 1))
    dynamic_model = DynamicModel.DynamicModelWrapper(cartpole, x_u, init_state, init_input, T)
    #################################
    ##### Objective Function ########
    #################################
    C_matrix_diag = sp.symbols("c:5")
    add_param_obj = np.zeros((T, 5), dtype = np.float64)
    for tau in range(T):
        if tau < T-1:
            add_param_obj[tau] = np.asarray((1, 0.1, 1, 1, 0.1))
        else: 
            add_param_obj[tau] = np.asarray((10000, 1000, 0, 0, 0))
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u)@np.diag(np.asarray(C_matrix_diag))@(x_u), x_u_var = x_u, add_param_var=C_matrix_diag, add_param=add_param_obj)
    #################################
    ########## Training #############
    #################################
    x0_u_lower_bound = [np.pi, -0, -0, -0, -15]
    x0_u_upper_bound = [np.pi,  0,  0,  0,  15]
    x0_u_bound = (x0_u_lower_bound, x0_u_upper_bound)
    dataset_train = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=trial_no)
    dataset_vali = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=10) 
    if network == "large":
        nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(LargeNetwork(n+m, n), init_state, init_input, T)
        nn_dynamic_model.pretrain(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_large_5.model")
    elif network == "small":
        nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(SmallNetwork(n+m, n), init_state, init_input, T)
        nn_dynamic_model.pretrain(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_small.model")
    elif network == "residual":
        nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(SmallResidualNetwork(n+m, n), init_state, init_input, T)
        nn_dynamic_model.pretrain(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_residual_5.model")

    #################################
    ######### iLQR Solver ###########
    #################################
    logger_id = Logger.loguru_start(file_name = file_name, 
                                    T=T, 
                                    trial_no = trial_no,  
                                    stopping_criterion = stopping_criterion,  
                                    max_iter = max_iter, 
                                    max_line_search = max_line_search,
                                    decay_rate = decay_rate, 
                                    decay_rate_max_iters = decay_rate_max_iters, 
                                    gaussian_filter_sigma = gaussian_filter_sigma, 
                                    gaussian_noise_sigma = gaussian_noise_sigma,
                                    init_state = init_state,
                                    obj_C = add_param_obj[0],
                                    obj_terminal_C = add_param_obj[-1],
                                    x0_u_lower_bound = x0_u_lower_bound,
                                    x0_u_upper_bound = x0_u_upper_bound,
                                    is_use_large_net = network)
    cartpole1_example = AdvancediLQR.NNiLQR(dynamic_model, objective_function)
    cartpole1_example.solve(file_name, nn_dynamic_model, dataset_train,
                            re_train_stopping_criterion=stopping_criterion, 
                            max_iter=max_iter,
                            max_line_search = max_line_search,
                            decay_rate=decay_rate,
                            decay_rate_max_iters=decay_rate_max_iters,
                            gaussian_filter_sigma = gaussian_filter_sigma,
                            gaussian_noise_sigma = gaussian_noise_sigma)
    Logger.loguru_end(logger_id)

def cartpole1_net_iLQR( T = 300, 
                        trial_no = 100,
                        stopping_criterion = 1e-4,
                        max_iter=100,
                        decay_rate=0.99,
                        decay_rate_max_iters=300,
                        gaussian_filter_sigma = 10,
                        gaussian_noise_sigma = 1,
                        network = "small"):
    file_name = "cartpole1_net_iLQR_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        #################################
    ######### Dynamic Model #########
    #################################
    cartpole, x_u, n, m = DynamicModel.cart_pole()
    init_state = np.asarray([np.pi, 0, 0, 0],dtype=np.float64).reshape(-1,1)
    init_input = np.zeros((T, m, 1))
    dynamic_model = DynamicModel.DynamicModelWrapper(cartpole, x_u, init_state, init_input, T)
    #################################
    ##### Objective Function ########
    #################################
    C_matrix_diag = sp.symbols("c:5")
    add_param_obj = np.zeros((T, 5), dtype = np.float64)
    for tau in range(T):
        if tau < T-1:
            add_param_obj[tau] = np.asarray((1, 0.1, 1, 1, 0.1))
        else: 
            add_param_obj[tau] = np.asarray((10000, 1000, 0, 0, 0))
    objective_function = ObjectiveFunction.ObjectiveFunctionWrapper((x_u)@np.diag(np.asarray(C_matrix_diag))@(x_u), x_u_var = x_u, add_param_var=C_matrix_diag, add_param=add_param_obj)
    #################################
    ########## Training #############
    #################################
    x0_u_lower_bound = [np.pi, -0, -0, -0, -15]
    x0_u_upper_bound = [np.pi,  0,  0,  0,  15]
    x0_u_bound = (x0_u_lower_bound, x0_u_upper_bound)
    dataset_train = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=trial_no)
    dataset_vali = DynamicModel.DynamicModelDataSetWrapper(dynamic_model, x0_u_bound, Trial_No=10) 
    if network == "large":
        nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(LargeNetwork(n+m, n), init_state, init_input, T)
        nn_dynamic_model.pretrain(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_neural_large.model")
    elif network == "small":
        nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(SmallNetwork(n+m, n), init_state, init_input, T)
        nn_dynamic_model.pretrain(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_neural_small.model")
    elif network == "small_residual":
        nn_dynamic_model = DynamicModel.NeuralDynamicModelWrapper(SmallResidualNetwork(n+m, n), init_state, init_input, T)
        nn_dynamic_model.pretrain(dataset_train, dataset_vali, max_epoch = 100000, stopping_criterion=stopping_criterion, lr = 0.001, model_name = "cartpole1_neural_small_residual.model")
    #################################
    ######### iLQR Solver ###########
    #################################
    logger_id = Logger.loguru_start(   file_name = file_name, 
                                            T=T, 
                                            trial_no = trial_no,  
                                            stopping_criterion = stopping_criterion,  
                                            max_iter = max_iter, 
                                            decay_rate = decay_rate, 
                                            decay_rate_max_iters = decay_rate_max_iters, 
                                            gaussian_filter_sigma = gaussian_filter_sigma, 
                                            gaussian_noise_sigma = gaussian_noise_sigma,
                                            init_state = init_state,
                                            obj_C = add_param_obj[0],
                                            obj_terminal_C = add_param_obj[-1],
                                            x0_u_lower_bound = x0_u_lower_bound,
                                            x0_u_upper_bound = x0_u_upper_bound,
                                            network = network)
    cartpole1_example = AdvancediLQR.NetiLQR(dynamic_model, objective_function)
    cartpole1_example.solve(file_name, nn_dynamic_model, dataset_train,
                                    re_train_stopping_criterion=stopping_criterion, 
                                    max_iter=max_iter,
                                    decay_rate=decay_rate,
                                    decay_rate_max_iters=decay_rate_max_iters,
                                    gaussian_filter_sigma = gaussian_filter_sigma,
                                    gaussian_noise_sigma = gaussian_noise_sigma)
    Logger.loguru_end(logger_id)

# %%
if __name__ == "__main__":
    # cartpole1_vanilla(T = 150, max_iter=500, is_check_stop = False)

    cartpole1_dd_iLQR(  T = 150,
                        trial_no=100,
                        stopping_criterion = 1e-4,
                        max_iter=1000,
                        max_line_search = 5,
                        decay_rate=0.98,
                        decay_rate_max_iters=200,
                        gaussian_filter_sigma = 5,
                        gaussian_noise_sigma = 1,
                        network = "large")

    # cartpole1_net_iLQR( T = 150,
    #                     trial_no=100,
    #                     stopping_criterion = 1e-4,
    #                     max_iter=1000,
    #                     decay_rate=0.98,
    #                     decay_rate_max_iters=200,
    #                     gaussian_filter_sigma = 5,
    #                     gaussian_noise_sigma = 1,
    #                     network = "small")
# %%
