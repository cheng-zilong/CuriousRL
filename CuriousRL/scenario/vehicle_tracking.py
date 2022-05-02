import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelBase
from CuriousRL.utils.Logger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class VehicleTracking(DynamicModelBase):
    """ In this example, the vehicle packing at 0, 0, heading to the postive direction of the y axis\\
        We hope the vechile can tracking the reference y=-10 with the velocity 8, and head to the right\\
        x0: position_x, x1: position_y, x2: heading anglue, x3: velocity, x4: steering angle, x5: acceleration\\
        If is_with_constraints = True, then the steering angle is limited to [-0.6, 0.6], acceleration is limited to [-3, 3]
    """
    def __init__(self, is_with_constraints = True, T = 150):
        ##### Dynamic Function ########
        n, m = 4, 2 # number of state = 4, number of action = 1, prediction horizon = 150
        h_constant = 0.1 # sampling time
        x_u_var = sp.symbols('x_u:6')
        d_constant = 3
        h_d_constanT = h_constant/d_constant
        b_function = d_constant \
                    + h_constant*x_u_var[3]*sp.cos(x_u_var[4]) \
                    -sp.sqrt(d_constant**2 - (h_constant**2)*(x_u_var[3]**2)*(sp.sin(x_u_var[4])**2))
        dynamic_function = sp.Array([ 
                    x_u_var[0] + b_function*sp.cos(x_u_var[2]), 
                    x_u_var[1] + b_function*sp.sin(x_u_var[2]), 
                    x_u_var[2] + sp.asin(h_d_constanT*x_u_var[3] * sp.sin(x_u_var[4])), 
                    x_u_var[3] + h_constant*x_u_var[5] 
                ]) 
        init_state = np.asarray([0,0,np.pi/2,0],dtype=np.float64).reshape(-1,1) 
        init_action = np.zeros((T,m,1)) 
        if is_with_constraints: 
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-0.6, 0.6], [-3, 3]]) 
        else:
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
        ##### Objective Function ########
        C_matrix = np.diag([0.,1.,1.,1.,10.,10.])
        r_vector = np.asarray([0.,-10.,0.,8.,0.,0.])
        obj_fun = (x_u_var - r_vector)@C_matrix@(x_u_var - r_vector)
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            box_constr = box_constr, 
                            init_state = init_state, 
                            init_action = init_action, 
                            obj_fun = obj_fun)

    def play(self, logger_folder=None, no_iter = -1):
        """ If logger_folder exists and the result file is saved, then the specific iteration can be chosen to play the animation. \\

            Parameter
            ----------
            logger_folder : string
                The name of the logger folder
            no_iter : int
                The number of iteration to play the animation
        """
        fig, ax = super().create_plot(figsize=(8, 2), xlim=(-5,75), ylim=(-15,5))
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        car = patches.FancyBboxPatch((0, 0), 3, 2, "round,pad=0.02")
        car.set_color('C0')
        ax.add_patch(car)
        plt.plot(trajectory[:,0], trajectory[:,1])
        self._is_interrupted=False
        for i in range(self.T):
            angle = trajectory[i,2,0]
            t_start = ax.transData
            x = trajectory[i,0,0] + 1*np.sin(angle)
            y = trajectory[i,1,0] - 1*np.cos(angle)
            rotate_center = t_start.transform([x, y])
            car.set_x(x)
            car.set_y(y)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], angle)
            t_end = t_start + t
            car.set_transform(t_end)
            fig.canvas.draw()
            plt.pause(0.01)
            if self._is_interrupted:
                return
        self._is_interrupted = True

class VehicleTrackingObs(DynamicModelBase):
    def __init__(self, is_with_constraints = True, T = 150):
        ##### Dynamic Function ########
        n, m = 4, 2 # number of state = 4, number of action = 1, prediction horizon = 150
        h_constant = 0.1 # sampling time
        x_u_var = sp.symbols('x_u:6')
        d_constant = 3
        h_d_constanT = h_constant/d_constant
        b_function = d_constant \
                    + h_constant*x_u_var[3]*sp.cos(x_u_var[4]) \
                    -sp.sqrt(d_constant**2 - (h_constant**2)*(x_u_var[3]**2)*(sp.sin(x_u_var[4])**2))
        dynamic_function = sp.Array([ 
                    x_u_var[0] + b_function*sp.cos(x_u_var[2]), 
                    x_u_var[1] + b_function*sp.sin(x_u_var[2]), 
                    x_u_var[2] + sp.asin(h_d_constanT*x_u_var[3] * sp.sin(x_u_var[4])), 
                    x_u_var[3] + h_constant*x_u_var[5] 
                ]) 
        init_state = np.asarray([0,0,np.pi/2,0],dtype=np.float64).reshape(-1,1) 
        init_action = np.zeros((T,m,1)) 
        if is_with_constraints: 
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-0.6, 0.6], [-3, 3]]) 
            other_constr = [-((x_u_var[0] - 20)**2 + (x_u_var[1] + 10)**2 - 16)]
        else:
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
            other_constr = []
        ##### Objective Function ########
        C_matrix = np.diag([0.,1.,1.,1.,10.,10.])
        r_vector = np.asarray([0.,-10.,0.,8.,0.,0.])
        obj_fun = (x_u_var - r_vector)@C_matrix@(x_u_var - r_vector)
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            box_constr = box_constr, 
                            other_constr = other_constr,
                            init_state = init_state, 
                            init_action = init_action, 
                            obj_fun = obj_fun)

    def play(self, logger_folder=None, no_iter = -1):
        fig, ax = super().create_plot(figsize=(8, 2), xlim=(-5,75), ylim=(-15,5))
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        car = patches.FancyBboxPatch((0, 0), 3, 2, "round,pad=0.02")
        car.set_color('C0')
        ax.add_patch(car)
        plt.plot(trajectory[:,0], trajectory[:,1])
        self._is_interrupted=False
        for i in range(self.T):
            angle = trajectory[i,2,0]
            t_start = ax.transData
            x = trajectory[i,0,0] + 1*np.sin(angle)
            y = trajectory[i,1,0] - 1*np.cos(angle)
            rotate_center = t_start.transform([x, y])
            car.set_x(x)
            car.set_y(y)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], angle)
            t_end = t_start + t
            car.set_transform(t_end)
            fig.canvas.draw()
            plt.pause(0.01)
            if self._is_interrupted:
                return
        self._is_interrupted = True

class VehicleTrackingADMMiLQR1(DynamicModelBase):
    def __init__(self, is_with_constraints = True, T = 60):
        ##### Dynamic Function ########
        n, m = 6, 2 
        x_u_var = sp.symbols('x_u:8')
        l_f = 1.06
        l_r = 1.85
        k_f = -128916
        k_r = -85944
        m_kg = 1412
        I_z = 1536.7
        T_s = 0.1
        dynamic_function = sp.Array([ 
            x_u_var[0]+T_s*(x_u_var[3]*sp.cos(x_u_var[2]) - x_u_var[4]*sp.sin(x_u_var[2])),
            x_u_var[1]+T_s*(x_u_var[4]*sp.cos(x_u_var[2]) + x_u_var[3]*sp.sin(x_u_var[2])),
            x_u_var[2]+T_s*x_u_var[5],
            x_u_var[3]+T_s*x_u_var[7],
            (m_kg*x_u_var[3]*x_u_var[4] + T_s*(l_f*k_f-l_r*k_r)*x_u_var[5]-T_s*k_f*x_u_var[6]*x_u_var[3]-T_s*m_kg*x_u_var[3]*x_u_var[3]*x_u_var[5])/(m_kg*x_u_var[3]-T_s*(k_f+k_r)),
            (I_z*x_u_var[3]*x_u_var[5] + T_s*(l_f*k_f-l_r*k_r)*x_u_var[4]-T_s*l_f*k_f*x_u_var[6]*x_u_var[3])/(I_z*x_u_var[3]-T_s*(l_f*l_f*k_f+l_r*l_r*k_r))
        ]) 
        init_state = np.asarray([0,0,0,0,0,0],dtype=np.float64).reshape(-1,1) 
        init_action = np.zeros((T,m,1)) 
        if is_with_constraints: 
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-0.6, 0.6], [-1.5, 3]]) 
            other_constr = [-(((x_u_var[0] - 15)**2)/25 + ((x_u_var[1] + 1)**2)/6.25 - 1)]
        else:
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
            other_constr = []
        ##### Objective Function ########
        C_matrix = np.diag([0.,1.,0.,1.,0.,0.,1.,1.])
        r_vector = np.asarray([0.,0.,0.,8.,0.,0.,0.,0.])
        obj_fun = (x_u_var - r_vector)@C_matrix@(x_u_var - r_vector)
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            box_constr = box_constr, 
                            other_constr = other_constr,
                            init_state = init_state, 
                            init_action = init_action, 
                            obj_fun = obj_fun)

    def play(self, logger_folder=None, no_iter = -1):
        fig, ax = super().create_plot(figsize=(8, 2), xlim=(-5,75), ylim=(-15,5))
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        car = patches.FancyBboxPatch((0, 0), 3, 2, "round,pad=0.02")
        car.set_color('C0')
        ax.add_patch(car)
        plt.plot(trajectory[:,0], trajectory[:,1])
        self._is_interrupted=False
        for i in range(self.T):
            angle = trajectory[i,2,0]
            t_start = ax.transData
            x = trajectory[i,0,0] + 1*np.sin(angle)
            y = trajectory[i,1,0] - 1*np.cos(angle)
            rotate_center = t_start.transform([x, y])
            car.set_x(x)
            car.set_y(y)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], angle)
            t_end = t_start + t
            car.set_transform(t_end)
            fig.canvas.draw()
            plt.pause(0.01)
            if self._is_interrupted:
                return
        self._is_interrupted = True

class VehicleTrackingADMMiLQR2(DynamicModelBase):
    def __init__(self, is_with_constraints = True, T = 60):
        ##### Dynamic Function ########
        n, m = 6, 2 
        x_u_var = sp.symbols('x_u:8')
        l_f = 1.06
        l_r = 1.85
        k_f = -128916
        k_r = -85944
        m_kg = 1412
        I_z = 1536.7
        T_s = 0.1
        dynamic_function = sp.Array([ 
            x_u_var[0]+T_s*(x_u_var[3]*sp.cos(x_u_var[2]) - x_u_var[4]*sp.sin(x_u_var[2])),
            x_u_var[1]+T_s*(x_u_var[4]*sp.cos(x_u_var[2]) + x_u_var[3]*sp.sin(x_u_var[2])),
            x_u_var[2]+T_s*x_u_var[5],
            x_u_var[3]+T_s*x_u_var[7],
            (m_kg*x_u_var[3]*x_u_var[4] + T_s*(l_f*k_f-l_r*k_r)*x_u_var[5]-T_s*k_f*x_u_var[6]*x_u_var[3]-T_s*m_kg*x_u_var[3]*x_u_var[3]*x_u_var[5])/(m_kg*x_u_var[3]-T_s*(k_f+k_r)),
            (I_z*x_u_var[3]*x_u_var[5] + T_s*(l_f*k_f-l_r*k_r)*x_u_var[4]-T_s*l_f*k_f*x_u_var[6]*x_u_var[3])/(I_z*x_u_var[3]-T_s*(l_f*l_f*k_f+l_r*l_r*k_r))
        ]) 
        init_state = np.asarray([0,0,0,0,0,0],dtype=np.float64).reshape(-1,1) 
        init_action = np.zeros((T,m,1)) 
        position_var = sp.symbols("p:2") # x and y
        add_param = np.hstack([np.linspace(0, 0.6*T, T).reshape(-1,1), 20+np.linspace(0, 0.3*T, T).reshape(-1,1)])
        if is_with_constraints: 
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-0.6, 0.6], [-1.5, 3]]) 
            other_constr = [-(((x_u_var[0] - position_var[0])**2)/25 + ((x_u_var[1] - 4)**2)/6.25 - 1), -(((x_u_var[0] - position_var[1])**2)/25 + ((x_u_var[1])**2)/6.25 - 1)]
        else:
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
            other_constr = []
        ##### Objective Function ########
        C_matrix = np.diag([0.,1.,0.,1.,0.,0.,1.,1.])
        r_vector = np.asarray([0.,4.,0.,8.,0.,0.,0.,0.])
        obj_fun = (x_u_var - r_vector)@C_matrix@(x_u_var - r_vector)
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            box_constr = box_constr, 
                            other_constr = other_constr,
                            init_state = init_state, 
                            init_action = init_action, 
                            obj_fun = obj_fun,
                            add_param_var= position_var,
                            add_param= add_param)

    def play(self, logger_folder=None, no_iter = -1):
        fig, ax = super().create_plot(figsize=(8, 2), xlim=(-5,75), ylim=(-15,5))
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        car = patches.FancyBboxPatch((0, 0), 3, 2, "round,pad=0.02")
        car.set_color('C0')
        ax.add_patch(car)
        plt.plot(trajectory[:,0], trajectory[:,1])
        self._is_interrupted=False
        for i in range(self.T):
            angle = trajectory[i,2,0]
            t_start = ax.transData
            x = trajectory[i,0,0] + 1*np.sin(angle)
            y = trajectory[i,1,0] - 1*np.cos(angle)
            rotate_center = t_start.transform([x, y])
            car.set_x(x)
            car.set_y(y)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], angle)
            t_end = t_start + t
            car.set_transform(t_end)
            fig.canvas.draw()
            plt.pause(0.01)
            if self._is_interrupted:
                return
        self._is_interrupted = True

class VehicleTrackingADMMiLQR3(DynamicModelBase):
    def __init__(self, is_with_constraints = True, T = 60):
        ##### Dynamic Function ########
        n, m = 6, 2 
        x_u_var = sp.symbols('x_u:8')
        l_f = 1.06
        l_r = 1.85
        k_f = -128916
        k_r = -85944
        m_kg = 1412
        I_z = 1536.7
        T_s = 0.1
        dynamic_function = sp.Array([ 
            x_u_var[0]+T_s*(x_u_var[3]*sp.cos(x_u_var[2]) - x_u_var[4]*sp.sin(x_u_var[2])),
            x_u_var[1]+T_s*(x_u_var[4]*sp.cos(x_u_var[2]) + x_u_var[3]*sp.sin(x_u_var[2])),
            x_u_var[2]+T_s*x_u_var[5],
            x_u_var[3]+T_s*x_u_var[7],
            (m_kg*x_u_var[3]*x_u_var[4] + T_s*(l_f*k_f-l_r*k_r)*x_u_var[5]-T_s*k_f*x_u_var[6]*x_u_var[3]-T_s*m_kg*x_u_var[3]*x_u_var[3]*x_u_var[5])/(m_kg*x_u_var[3]-T_s*(k_f+k_r)),
            (I_z*x_u_var[3]*x_u_var[5] + T_s*(l_f*k_f-l_r*k_r)*x_u_var[4]-T_s*l_f*k_f*x_u_var[6]*x_u_var[3])/(I_z*x_u_var[3]-T_s*(l_f*l_f*k_f+l_r*l_r*k_r))
        ]) 
        init_state = np.asarray([0,0,0,4,0,0],dtype=np.float64).reshape(-1,1) 
        init_action = np.zeros((T,m,1)) 
        position_var = sp.symbols("p:2") # x and y
        tau = np.linspace(0, 0.6*T, T).reshape(-1,1)
        add_param = np.hstack([30-(0.2/3)*(0.1*tau)**3+(0.1*tau)**2+3*0.1*tau, 10+np.linspace(0, 1*T, T).reshape(-1,1)])
        if is_with_constraints: 
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-0.6, 0.6], [-1.5, 3]]) 
            other_constr = [-(((x_u_var[0] - position_var[0])**2)/25 + ((x_u_var[1]+0.2)**2)/6.25 - 1), -(((x_u_var[0] - position_var[1])**2)/25 + ((x_u_var[1]-4)**2)/6.25 - 1)]
        else:
            box_constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
            other_constr = []
        ##### Objective Function ########
        C_matrix = np.diag([0.,1.,0.,1.,0.,0.,1.,1.])
        r_vector = np.asarray([0.,0.,0.,15.,0.,0.,0.,0.])
        obj_fun = (x_u_var - r_vector)@C_matrix@(x_u_var - r_vector)
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            box_constr = box_constr, 
                            other_constr = other_constr,
                            init_state = init_state, 
                            init_action = init_action, 
                            obj_fun = obj_fun,
                            add_param_var= position_var,
                            add_param= add_param)

    def play(self, logger_folder=None, no_iter = -1):
        fig, ax = super().create_plot(figsize=(8, 2), xlim=(-5,75), ylim=(-15,5))
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        car = patches.FancyBboxPatch((0, 0), 3, 2, "round,pad=0.02")
        car.set_color('C0')
        ax.add_patch(car)
        plt.plot(trajectory[:,0], trajectory[:,1])
        self._is_interrupted=False
        for i in range(self.T):
            angle = trajectory[i,2,0]
            t_start = ax.transData
            x = trajectory[i,0,0] + 1*np.sin(angle)
            y = trajectory[i,1,0] - 1*np.cos(angle)
            rotate_center = t_start.transform([x, y])
            car.set_x(x)
            car.set_y(y)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], angle)
            t_end = t_start + t
            car.set_transform(t_end)
            fig.canvas.draw()
            plt.pause(0.01)
            if self._is_interrupted:
                return
        self._is_interrupted = True