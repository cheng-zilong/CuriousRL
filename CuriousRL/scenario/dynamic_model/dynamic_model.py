#%%
import numpy as np
import sympy as sp
import scipy as sci
import time as tm
from scipy import io
from numba import njit
import torch
from torch import nn
import torch_optimizer as optim
from torch.autograd.functional import jacobian
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from CuriousRL.utils.Logger import logger
from CuriousRL.scenario import ScenarioWrapper


class DynamicModelWrapper(ScenarioWrapper):
    """ This is a wrapper class for the dynamic model
    """
    def __init__(self, 
        algo, 
        dynamic_model_function, 
        x_u_var, 
        init_state, 
        init_input_traj, 
        T, 
        add_param_var = None, 
        add_param = None
        ):
        """ Initialization
            
            Parameters
            ---------------
            dynamic_model_function : sympy.array with symbols
                The model dynamic function defined by sympy symbolic array
            x_u_var : tuple with sympy.symbols 
                State and input variables in the model
            init_state : array(n, 1)
                The initial state vector of the system
            init_input_traj : array(T, m, 1) 
                The initial input vector
            T : int
                The prediction horizon
            add_param_var : tuple with sympy.symbols 
                Introduce the additional variables that are not derived
            add_param : array(T, -1)
                Give the values to the additioanl variables
        """
        super.__init__(algo = algo)
        self.init_state = init_state
        self.init_input_traj = init_input_traj
        self.n = int(init_state.shape[0])
        self.m = int(len(x_u_var) - self.n)
        self.T = T
        if add_param_var is None:
            add_param_var = sp.symbols("no_use")
        self.dynamic_model_lamdify = njit(sp.lambdify([x_u_var, add_param_var], dynamic_model_function, "math"))
        grad_dynamic_model_function = sp.transpose(sp.derive_by_array(dynamic_model_function, x_u_var))
        self.grad_dynamic_model_lamdify = njit(sp.lambdify([x_u_var, add_param_var], grad_dynamic_model_function, "math"))
        self.add_param = add_param
        
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
            input_traj = self.init_input_traj
        return self._eval_traj_static(self.dynamic_model_lamdify, init_state, input_traj, self.add_param, self.m, self.n)

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
        return self._update_traj_static(self.dynamic_model_lamdify, self.m, self.n, old_traj, K_matrix_all, k_vector_all, alpha, self.add_param)

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
        return self._eval_grad_dynamic_model_static(self.grad_dynamic_model_lamdify, trajectory, self.add_param)
    
    @staticmethod
    @njit
    def _eval_traj_static(dynamic_model_lamdify, init_state, input_traj, add_param, m, n):
        T = int(input_traj.shape[0])
        if add_param == None:
            add_param = np.zeros((T,1))
        trajectory = np.zeros((T, m+n, 1))
        trajectory[0] = np.vstack((init_state, input_traj[0]))
        for tau in range(T-1):
            trajectory[tau+1, :n, 0] = np.asarray(dynamic_model_lamdify(trajectory[tau,:,0], add_param[tau]),dtype=np.float64)
            trajectory[tau+1, n:] = input_traj[tau+1]
        return trajectory

    @staticmethod
    @njit
    def _update_traj_static(dynamic_model_lamdify, m, n, old_traj, K_matrix_all, k_vector_all, alpha, add_param):
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
            new_trajectory[tau+1,0:n] = np.asarray(dynamic_model_lamdify(new_trajectory[tau,:,0], add_param[tau]),dtype=np.float64).reshape(-1,1)
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

class NeuralDynamicModelWrapper(DynamicModelWrapper):
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


# %%
