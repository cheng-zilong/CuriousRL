import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelWrapper
from CuriousRL.utils.Logger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class CarParking(DynamicModelWrapper):
    """ In this example, the vehicle packing at 0, 0, heading to the top\\
        We hope the vechile can pack at 0, 0, and head to the right\\
        x0: position_x, x1: position_y, x2: heading anglue, x3: velocity, x4: steering angle, x5: acceleration\\
        If is_with_constraints = True, then the steering angle is limited to [-0.5, 0.5], acceleration is limited to [-2, 2]
    """
    def __init__(self, is_with_constraints = True, T = 500):
        ##### Dynamic Function ########
        n, m = 4, 2 # number of state = 4, number of input = 1, prediction horizon = 150
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
                    x_u_var[2] + sp.asin(h_d_constanT*x_u_var[3]*sp.sin(x_u_var[4])), 
                    x_u_var[3]+h_constant*x_u_var[5]
                ])
        init_state = np.asarray([1,-1,np.pi/2,0],dtype=np.float64).reshape(-1,1)
        init_input = np.zeros((T,m,1))
        if is_with_constraints: 
            constr = np.asarray([[-1, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-0.6, 0.6], [-3, 3]]) 
        else:
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
        ##### Objective Function ########
        switch_var = sp.symbols("s:2")
        add_param_obj = np.zeros((T, 2), dtype = np.float64)
        for tau in range(T):
            if tau < T-1:
                add_param_obj[tau] = np.asarray((1.,0.))
            else: 
                add_param_obj[tau] = np.asarray((0.,1.))

        def Huber_fun(x, p):
            return sp.sqrt((x**2)+(p**2)) - p
        runing_obj = Huber_fun(x_u_var[0], 0.01) + Huber_fun(x_u_var[1], 0.01) + Huber_fun(x_u_var[2], 0.01)
        terminal_obj = 1000*Huber_fun(x_u_var[0], 0.01) + 1000*Huber_fun(x_u_var[1], 0.01) + 1000*Huber_fun(x_u_var[2], 0.01) + 100*Huber_fun(x_u_var[3], 0.01)
        input_obj = x_u_var[4]**2 + x_u_var[5]**2
        obj_fun = switch_var[0] * runing_obj + switch_var[1]*terminal_obj + input_obj
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            constr = constr, 
                            init_state = init_state, 
                            init_input = init_input, 
                            obj_fun = obj_fun,
                            add_param_var=switch_var,
                            add_param=add_param_obj)

    def play(self, logger_folder=None, no_iter = -1):
        """ If logger_folder exists and the result file is saved, then the specific iteration can be chosen to play the animation. \\

            Parameter
            ----------
            logger_folder : string
                The name of the logger folder
            no_iter : int
                The number of iteration to play the animation
        """
        fig, ax = super().create_plot(figsize=(5, 5), xlim=(-5,10), ylim=(-7.5,7.5))
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        car = patches.FancyBboxPatch((0, 0), 3, 2, "round,pad=0.2")
        car.set_color('C0')
        ax.add_patch(car)
        plt.plot(trajectory[:,0], trajectory[:,1],'C1')
        plt.plot([-1,-1], [10,-10],'C2')
        plt.plot([-1,5], [2,2],'C2')
        plt.plot([-1,5], [-2,-2],'C2')
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
            plt.pause(0.001)
            if self._is_interrupted:
                return
        self._is_interrupted = True
