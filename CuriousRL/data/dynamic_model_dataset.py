class DynamicModelDataSetWrapper(object):
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
        self.dataset_x = torch.zeros((self.Trial_No, self.T-1, self.n+self.m,1)).cuda()
        self.dataset_y = torch.zeros((self.Trial_No, self.T-1, self.n,1)).cuda()
        # The index of the dataset for the next time updating
        self.update_index = 0
        x0_u_lower_bound, x0_u_upper_bound = x0_u_bound
        for i in range(self.Trial_No):
            x0 = np.random.uniform(x0_u_lower_bound[:self.n], x0_u_upper_bound[:self.n]).reshape(-1,1)
            input_trajectory = np.expand_dims(np.random.uniform(x0_u_lower_bound[self.n:], x0_u_upper_bound[self.n:], (self.T, self.m)), axis=2)
            new_trajectory = torch.from_numpy(dynamic_model.eval_traj(x0, input_trajectory)).float().cuda()
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
                self.dataset_x[self.update_index] = torch.from_numpy(trajectory[:self.T-1]).float().cuda()
                self.dataset_y[self.update_index] = torch.from_numpy(trajectory[1:,:self.n]).float().cuda()
                if self.update_index < self.Trial_No - 1:
                    self.update_index = self.update_index + 1
                else:
                    self.update_index  = 0
        else:
            self.dataset_x[self.update_index] = torch.from_numpy(new_trajectory[:self.T-1]).float().cuda()
            self.dataset_y[self.update_index] = torch.from_numpy(new_trajectory[1:,:self.n]).float().cuda()
            if self.update_index < self.Trial_No - 1:
                self.update_index = self.update_index + 1
            else:
                self.update_index  = 0
        logger.debug("[+ +] Dataset is updated!")
    def get_data(self):
        """ Return the data from the dataset

            Return
            ---------
            X : tensor(dataset_size, n+m)\\
            Y : tensor(dataset_size, n)
        """
        return self.X, self. Y