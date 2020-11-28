#%%
from __future__ import annotations
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
import os
import torch
from .basic_ilqr import iLQRWrapper
from CuriousRL.scenario.dynamic_model.dynamic_model import DynamicModelWrapper
from CuriousRL.data.dataset_wrapper import DatasetWrapper
from loguru import logger
from scipy.ndimage import gaussian_filter1d


class DynamicModelDataset(DatasetWrapper):
    """This class generates the dynamic model data for training neural networks
    """

    def _init_dataset(self, dynamic_model, x0_u_bound):
        """ Inner method, initialize the dataset of a system model

            Parameters
            ---------
            dynamic_model : DynamicModelWrapper
                The system generating data
            x0_u_bound : tuple (x0_u_lower_bound, x0_u_upper_bound)
                x0_u_lower_bound : list(m+n)
                x0_u_upper_bound : list(m+n)
                Since you are generating the data with random initial states and inputs, 
                you need to give
                the range of the initial system state variables, and
                the range of the system input variables
        """

        self.dataset_x = torch.zeros(
            (self.Trial_No, self.T-1, self.n+self.m, 1)).cuda()
        self.dataset_y = torch.zeros(
            (self.Trial_No, self.T-1, self.n, 1)).cuda()
        # The index of the dataset for the next time updating
        self.update_index = 0
        x0_u_lower_bound, x0_u_upper_bound = x0_u_bound
        for i in range(self.Trial_No):
            x0 = np.random.uniform(
                x0_u_lower_bound[:self.n], x0_u_upper_bound[:self.n]).reshape(-1, 1)
            input_trajectory = np.expand_dims(np.random.uniform(
                x0_u_lower_bound[self.n:], x0_u_upper_bound[self.n:], (self.T, self.m)), axis=2)
            new_trajectory = torch.from_numpy(
                dynamic_model.eval_traj(x0, input_trajectory)).float().cuda()
            self.dataset_x[i] = new_trajectory[:self.T-1]
            self.dataset_y[i] = new_trajectory[1:, :self.n]
        self.X = self.dataset_x.view(self.dataset_size, self.n+self.m)
        self.Y = self.dataset_y.view(self.dataset_size, self.n)

    def __init__(self, dynamic_model, x0_u_bound, Trial_No):
        """ Initialization

            Parameters
            ---------
            dynamic_model : DynamicModelWrapper
                The system generating data
            x0_u_bound : tuple (x0_u_lower_bound, x0_u_upper_bound)
                x0_u_lower_bound : list(m+n)
                x0_u_upper_bound : list(m+n)
                Since you are generating the data with random initial states and inputs, 
                you need to give
                the range of the initial system state variables, and
                the range of the system input variables
            Trial_No : int
                The number of trials 
                The dataset size is Trial_No*(T-1)
        """
        self.Trial_No = Trial_No
        self.T = dynamic_model.T
        self.n = dynamic_model.n
        self.m = dynamic_model.m
        self.dataset_size = self.Trial_No*(self.T-1)
        self._init_dataset(dynamic_model, x0_u_bound)

    def update_dataset(self, new_trajectory):
        """ Insert new data to the dataset and delete the oldest data

            Parameter
            -------
            new_trajectory : array(T, n+m, 1)
                The new trajectory inserted in to the dataset
        """
        if isinstance(new_trajectory, list):
            for trajectory in new_trajectory:
                self.dataset_x[self.update_index] = torch.from_numpy(
                    trajectory[:self.T-1]).float().cuda()
                self.dataset_y[self.update_index] = torch.from_numpy(
                    trajectory[1:, :self.n]).float().cuda()
                if self.update_index < self.Trial_No - 1:
                    self.update_index = self.update_index + 1
                else:
                    self.update_index = 0
        else:
            self.dataset_x[self.update_index] = torch.from_numpy(
                new_trajectory[:self.T-1]).float().cuda()
            self.dataset_y[self.update_index] = torch.from_numpy(
                new_trajectory[1:, :self.n]).float().cuda()
            if self.update_index < self.Trial_No - 1:
                self.update_index = self.update_index + 1
            else:
                self.update_index = 0
        logger.debug("[+ +] Dataset is updated!")

    def get_data(self):
        """ Return the data from the dataset

            Return
            ---------
            X : tensor(dataset_size, n+m)\\
            Y : tensor(dataset_size, n)
        """
        return self.X, self. Y

class NNiLQRDynamicModel(iLQRDynamicModel):
    """ This is a class to create system model using neural network
    """
    def __init__(self, network, init_state, init_input_traj, T):
        """ Initialization
            network : nn.module
                Networks used to train the system dynamic model
            init_state : array(n,1)
                Initial system state
            init_input_traj : array(T, m, 1)
                Initial input trajectory used to generalize the initial trajectory
            T : int
                Prediction horizon
        """
        self.init_input_traj = init_input_traj
        self.init_state = init_state
        self.n = init_state.shape[0]
        self.m = init_input_traj.shape[1]
        self.T = T
        self.model = network.cuda()
        self.F_matrix_all = torch.zeros(self.T, self.n, self.n+self.m).cuda()
        self.const_param = torch.eye(self.n).cuda()

    def pretrain(self, dataset_train, dataset_vali, max_epoch=50000, stopping_criterion = 1e-3, lr = 1e-3, model_name = "NeuralDynamic.model"):
        """ Pre-train the model by using randomly generalized data

            Parameters
            ------------
            dataset_train : DynamicModelDataSetWrapper
                Data set for training
            dataset_vali : DynamicModelDataSetWrapper
                Data set for validation
            max_epoch : int
                Maximum number of epochs if stopping criterion is not reached
            stopping_criterion : double
                If the objective function of the training set is less than 
                the stopping criterion, the training is stopped
            lr : double
                Learning rate
            model_name : string
                When the stopping criterion, 
                the model with the given name will be saved as a file
        """
        # if the model exists, load the model directly
        model_path = os.path.join("models", model_name)
        result_train_loss = np.zeros(max_epoch) 
        result_vali_loss = np.zeros(int(max_epoch/100)) 

        if not os.path.exists(model_path):
            logger_id = logger.add(os.path.join("models", model_name + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"), format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - {message}")
            logger.debug("[+ +] Model file \"" + model_name + "\" does NOT exist. Pre-traning starts...")
            self.writer = SummaryWriter()
            loss_fun = nn.MSELoss()
            # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            optimizer = optim.RAdam(self.model.parameters(), lr=lr, weight_decay=1e-4)
            X_train, Y_train = dataset_train.get_data()
            X_vali, Y_vali = dataset_vali.get_data()  
            time_start_pretraining = tm.time()
            for epoch in range(max_epoch):
                #### Training ###
                self.model.train()
                optimizer.zero_grad()
                Y_prediction = self.model(X_train)         
                obj_train = loss_fun(Y_prediction, Y_train) 
                obj_train.backward()                   
                optimizer.step()
                result_train_loss[epoch] = obj_train.item()
                if obj_train.item() < stopping_criterion or epoch % 100 == 0: # Check stopping criterion
                    ## Evaluation ###
                    self.model.eval()
                    Y_prediction = self.model(X_vali)         # Forward Propagation
                    obj_vali = loss_fun(Y_prediction, Y_vali)
                    ##### Print #####
                    logger.debug("[+ +] Epoch: %5d     Train Loss: %.5e     Vali Loss:%.5e"%(
                            epoch + 1,      obj_train.item(),  obj_vali.item()))
                    self.writer.add_scalar('Loss/train', obj_train.item(), epoch)
                    self.writer.add_scalar('Loss/Vali', obj_vali.item(), epoch)
                    result_vali_loss[int(np.ceil(epoch/100))] = obj_vali.item()
                    if obj_train.item() < stopping_criterion:
                        time_end_preraining = tm.time()
                        time_pretraining = time_end_preraining - time_start_pretraining
                        logger.debug("[+ +] Pretraining finished! Model file \"" + model_name + "\" is saved!")
                        logger.debug("[+ +] Pretraining time: %.8f"%(time_pretraining))
                        torch.save(self.model.state_dict(), model_path)     
                        io.savemat(os.path.join("models", model_name + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".mat"), {"Train_loss": result_train_loss, "Vali_loss": result_vali_loss})
                        logger.remove(logger_id)
                        return
            raise Exception("Maximum epoch is reached!")
        else:
            logger.debug("[+ +] Model file \"" + model_name + "\" exists. Loading....")
            self.model.load_state_dict(torch.load(model_path))
            logger.debug("[+ +] Loading Completed!")
            self.model.eval()

    def retrain(self, dataset, max_epoch=10000, stopping_criterion = 1e-3, lr = 0.001):
        logger.debug("[+ +] Re-traning starts...")
        loss_fun = nn.MSELoss()
        # optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.RAdam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        X_train, Y_train = dataset.get_data()
        for epoch in range(max_epoch):
            #### Training ###
            self.model.train()
            optimizer.zero_grad()
            Y_prediction = self.model(X_train)         
            obj_train = loss_fun(Y_prediction, Y_train)  
            obj_train.backward()             
            optimizer.step()
            if obj_train.item() < stopping_criterion or epoch % 100 == 0:  # Check stopping criterion
                logger.debug("[+ +] Epoch: %5d   Train Obj: %.5e"%(
                                    epoch + 1,     obj_train.item()))
                if obj_train.item() < stopping_criterion:
                    logger.debug("[+ +] Re-training finished!")
                    self.model.eval()
                    return      
        raise Exception("Maximum epoch is reached!")

    def next_state(self, current_state_and_input):
        if isinstance(current_state_and_input, list):
            current_state_and_input = np.asarray(current_state_and_input)
        if current_state_and_input.shape[0] != 1:
            current_state_and_input = current_state_and_input.reshape(1,-1)
        x_u = torch.from_numpy(current_state_and_input).float().cuda()
        with torch.no_grad():
            return self.model(x_u).numpy().reshape(-1,1)

    def eval_traj(self, init_state=None, input_traj=None):
        if init_state is None:
            init_state = self.init_state
        if input_traj is None:
            input_traj = self.init_input_traj
        input_trajectory_cuda = torch.from_numpy(input_traj).float().cuda()
        trajectory = torch.zeros(self.T, self.n+self.m).cuda()
        trajectory[0] = torch.from_numpy(np.vstack((init_state, input_traj[0]))).float().cuda().reshape(-1)
        with torch.no_grad():
            for tau in range(self.T-1):
                trajectory[tau+1, :self.n] = self.model(trajectory[tau,:].reshape(1,-1))
                trajectory[tau+1, self.n:] = input_trajectory_cuda[tau+1,0]
        return trajectory.cpu().double().numpy().reshape(self.T, self.m+self.n, 1)

    def update_traj(self, old_traj, K_matrix_all, k_vector_all, alpha): 
        new_traj = np.zeros((self.T, self.m+self.n, 1))
        new_traj[0] = old_traj[0] # initial states are the same
        for tau in range(self.T-1):
            # The amount of change of state x
            delta_x = new_traj[tau, :self.n] - old_traj[tau, :self.n]
            # The amount of change of input u
            delta_u = K_matrix_all[tau]@delta_x+alpha*k_vector_all[tau]
            # The real input of next iteration
            input_u = old_traj[tau, self.n:self.n+self.m] + delta_u
            new_traj[tau,self.n:] = input_u
            with torch.no_grad():
                new_traj[tau+1,0:self.n,0] = self.model(torch.from_numpy(new_traj[tau,:].T).float().cuda()).cpu().double().numpy()
            # dont care the input at the last time stamp, because it is always zero
        return new_traj

    def eval_grad_dynamic_model(self, trajectory):
        trajectory_cuda = torch.from_numpy(trajectory[:,:,0]).float().cuda()
        # def get_batch_jacobian(net, x, noutputs):
        #     x = x.unsqueeze(1) # b, 1 ,in_dim
        #     n = x.size()[0]
        #     x = x.repeat(1, noutputs, 1) # b, out_dim, in_dim
        #     x.requires_grad_(True)
        #     y = net(x)
        #     input_val = torch.eye(noutputs).reshape(1,noutputs, noutputs).repeat(n, 1, 1)
        #     y.backward(input_val)
        #     return x.grad.data
        for tau in range(0, self.T):
            x = trajectory_cuda[tau]
            x = x.repeat(self.n, 1)
            x.requires_grad_(True)
            y = self.model(x)
            y.backward(self.const_param)
            self.F_matrix_all[tau] = x.grad.data
            # F_matrix_list[tau] = jacobian(self.model, torch.from_numpy().float()).squeeze().numpy()
        # get_batch_jacobian(self.model, trajectory_cuda, 4)
        return self.F_matrix_all.cpu().double().numpy()

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
