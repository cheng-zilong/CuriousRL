from __future__ import annotations
import numpy as np
import time as tm
from scipy import io
import os
import torch
from torch import Tensor
from .basic_ilqr import iLQRWrapper
from CuriousRL.utils.Logger import logger
from scipy.ndimage import gaussian_filter1d
from .ilqr_dynamic_model import iLQRDynamicModel
if TYPE_CHECKING:
    from CuriousRL.data import Data
    from CuriousRL.data import Dataset

class NNiLQRDynamicModel(iLQRDynamicModel):
    """ NNiLQRDynamicModel uses a neural network to fit the dynamic model of a system. This algorithm can only be implemented on a cuda device.
    """
    def __init__(self, network, init_state, init_input):
        """ Initialization
            network : nn.module
                Networks used to train the system dynamic model
            init_state : array(n,1)
                Initial system state
            init_input_traj : array(T, m, 1)
                Initial input trajectory used to generalize the initial trajectory
        """
        self._init_input = init_input
        self._init_state = init_state
        self._n = init_state.shape[0]
        self._m = init_input.shape[1]
        self._T = int(init_input.shape[0])
        self._model = network.cuda()
        self._F_matrix = Tensor((self._T, self._n, self._m+self._n)).cuda()
        self.__constant1 = torch.eye(self._n).cuda()

    def process_data(self, dataset) -> Tuple[Tensor, Tensor]:
        data = dataset.fetch_all_data()
        obs = data.obs
        actions = data.action
        traj = torch.cat((obs, actions), 1)
        X = traj[0:-1]
        Y = traj[1:]
        return X, Y

    def pretrain(self, dataset_train:Dataset, dataset_vali:Dataset, max_epoch=50000, stopping_criterion = 1e-3, lr = 1e-3, model_name = "NeuralDynamic.model"):
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
        model_path = logger.logger_path
        result_train_loss = np.zeros(max_epoch) 
        result_vali_loss = np.zeros(int(max_epoch/100)) 

        if not os.path.exists(model_path):
            logger.info("[+ +] Model file \"" + model_name + "\" does NOT exist. Pre-traning starts...")
            loss_fun = nn.MSELoss()
            # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            optimizer = optim.RAdam(self._model.parameters(), lr=lr, weight_decay=1e-4)
            X_train, Y_train = self.process_data(dataset_train)
            X_vali, Y_vali = self.process_data(dataset_vali)
            time_start_pretraining = tm.time()
            for epoch in range(max_epoch):
                #### Training ###
                self._model.train()
                optimizer.zero_grad()
                Y_prediction = self._model(X_train)         
                obj_train = loss_fun(Y_prediction, Y_train) 
                obj_train.backward()                   
                optimizer.step()
                result_train_loss[epoch] = obj_train.item()
                if obj_train.item() < stopping_criterion or epoch % 100 == 0: # Check stopping criterion
                    ## Evaluation ###
                    self._model.eval()
                    Y_prediction = self._model(X_vali)         # Forward Propagation
                    obj_vali = loss_fun(Y_prediction, Y_vali)
                    ##### Print #####
                    logger.info("[+ +] Epoch: %5d     Train Loss: %.5e     Vali Loss:%.5e"%(
                                        epoch + 1,    obj_train.item(),    obj_vali.item()))
                    result_vali_loss[int(np.ceil(epoch/100))] = obj_vali.item()
                    if obj_train.item() < stopping_criterion:
                        time_end_preraining = tm.time()
                        time_pretraining = time_end_preraining - time_start_pretraining
                        logger.info("[+ +] Pretraining finished! Model file \"" + model_name + "\" is saved!")
                        logger.info("[+ +] Pretraining time: %.8f"%(time_pretraining))
                        torch.save(self._model.state_dict(), os.path.join(model_path, model_name))     
                        io.savemat(os.path.join(model_path, model_name + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".mat"), {"Train_loss": result_train_loss, "Vali_loss": result_vali_loss})
                        return
            raise Exception("Maximum epoch in the pretraining is reached!")
        else:
            logger.info("[+ +] Model file \"" + model_name + "\" exists. Loading....")
            self._model.load_state_dict(torch.load(os.path.join(model_path, model_name)))
            self._model.eval()

    def retrain(self, dataset, max_epoch=10000, stopping_criterion = 1e-3, lr = 1e-3):
        logger.info("[+ +] Re-traning starts...")
        loss_fun = nn.MSELoss()
        # optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.RAdam(self._model.parameters(), lr=lr, weight_decay=1e-4)
        X_train, Y_train = self.process_data(dataset)
        for epoch in range(max_epoch):
            #### Training ###
            self._model.train()
            optimizer.zero_grad()
            Y_prediction = self._model(X_train)         
            obj_train = loss_fun(Y_prediction, Y_train)  
            obj_train.backward()             
            optimizer.step()
            if obj_train.item() < stopping_criterion or epoch % 100 == 0:  # Check stopping criterion
                logger.info("[+ +] Epoch: %5d   Train Obj: %.5e"%(
                                    epoch + 1,     obj_train.item()))
                if obj_train.item() < stopping_criterion:
                    logger.info("[+ +] Re-training finished!")
                    self._model.eval()
                    return
        raise Exception("Maximum epoch in the retraining is reached!")

    def eval_traj(self, init_state = None, input_traj = None):
        if init_state is None:
            init_state = self._init_state
        if input_traj is None:
            input_traj = self._init_input
        input_traj_cuda = torch.from_numpy(input_traj).float().cuda()
        trajectory = torch.zeros(self._T, self._n+self._m).cuda()
        trajectory[0] = torch.from_numpy(np.vstack((init_state, input_traj[0]))).float().cuda().reshape(-1)
        with torch.no_grad():
            for tau in range(self._T-1):
                trajectory[tau+1, :self._n] = self._model(trajectory[tau,:].reshape(1,-1))
                trajectory[tau+1, self._n:] = input_traj_cuda[tau+1,0]
        return trajectory.cpu().double().numpy().reshape(self._T, self._m+self._n, 1)

    def update_traj(self, old_traj, K_matrix, k_vector, alpha): 
        new_traj = np.zeros((self._T, self._m+self._n, 1))
        new_traj[0] = old_traj[0] # initial states are the same
        for tau in range(self._T-1):
            # The amount of change of state x
            delta_x = new_traj[tau, :self._n] - old_traj[tau, :self._n]
            # The amount of change of input u
            delta_u = K_matrix[tau]@delta_x+alpha*k_vector[tau]
            # The real input of next iteration
            input_u = old_traj[tau, self._n:self._n+self._m] + delta_u
            new_traj[tau,self._n:] = input_u
            with torch.no_grad():
                new_traj[tau+1,0:self._n,0] = self._model(torch.from_numpy(new_traj[tau,:].T).float().cuda()).cpu().double().numpy()
            # dont care the input at the last time stamp, because it is always zero
        return new_traj

    def eval_grad_dynamic_model(self, trajectory):
        trajectory_cuda = torch.from_numpy(trajectory[:,:,0]).float().cuda()
        for tau in range(0, self._T):
            x = trajectory_cuda[tau]
            x = x.repeat(self._n, 1)
            x.requires_grad_(True)
            y = self._model(x)
            y.backward(self.__constant1)
            self._F_matrix[tau] = x.grad.data
        return self._F_matrix.cpu().double().numpy()

class NNiLQR(iLQRWrapper):
    """This is a Neural Network iLQR class
    """
    def __init__(self,
                max_iter=1000,
                is_check_stop=True,
                iLQR_stopping_criterion=1e-6,
                max_line_search=50,
                gamma=0.5,
                line_search_method="vanilla",
                stopping_method="relative",
                decay_rate = 1, 
                decay_rate_max_iters = 300,
                gaussian_filter_sigma = 10,
                gaussian_noise_sigma = 1):
        super().__init__(stopping_criterion=iLQR_stopping_criterion,
                         max_line_search=max_line_search,
                         gamma=gamma,
                         line_search_method=line_search_method,
                         stopping_method=stopping_method,
                         max_iter = max_iter,
                         is_check_stop = is_check_stop)
        self._max_iter = max_iter
        self._is_check_stop = is_check_stop


    def init(self, scenario):
        self._nn_dynamic_model = None
        self.dataset_train
        pass

    def solve(  self,
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
        logger.info("[+ +] Initial Obj.Val.: %.5e"%(self.get_obj_fun_value()))
        trajectory = self.get_traj()
        new_data = []
        result_obj_val = np.zeros(max_iter)
        result_iter_time = np.zeros(max_iter)
        for i in range(int(max_iter)):
            if i == 1:  # skip the compiling time 
                start_time = tm.time()
            iter_start_time = tm.time()
            F_matrix = self._nn_dynamic_model.eval_grad_dynamic_model(trajectory)
            F_matrix = gaussian_filter1d(F_matrix, sigma = gaussian_filter_sigma, axis=0)
            self.update_F_matrix(F_matrix)
            self.backward_pass()
            obj_val, isStop = self.forward_pass(max_line_search=max_line_search)
            if i < decay_rate_max_iters:
                re_train_stopping_criterion = re_train_stopping_criterion * decay_rate
            iter_end_time = tm.time()
            iter_time = iter_end_time-iter_start_time
            logger.info("[+ +] Iter.No.:%3d  Iter.Time:%.3e   Obj.Val.:%.5e"%(
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
                self.dataset_train.update_dataset(new_data[-int(self.dataset_train.Trial_No/5):]) # at most update 20% dataset
                result_path = os.path.join(logger.logger_path, str(i) +".mat")
                io.savemat(result_path,{"optimal_trajectory": self.get_traj(), "trajectroy_noisy": trajectory_noisy})
                self._nn_dynamic_model.retrain(self.dataset_train, max_epoch = 100000, stopping_criterion = re_train_stopping_criterion)
                new_data = []
            else: 
                new_data += [trajectory]
        end_time = tm.time()
        io.savemat(os.path.join(logger_path,  "_result.mat"),{"obj_val": result_obj_val, "iter_time": result_iter_time})
        logger.info("[+ +] Completed! All Time:%.5e"%(end_time-start_time))