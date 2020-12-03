#%%
import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelWrapper
from CuriousRL.utils.Logger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class CartPoleSwingUp2(DynamicModelWrapper):
    """ In this example, the cartpole system is static at 0, 0, heading to the postive direction of the y axis\\
        We hope the vechile can tracking the reference y=-10 with the velocity 8, and head to the right\\
        x0: position, x1: velocity, x2: sin(angle), x3: cos(angle), x4: angular velocity, x5: force
        If is_with_constraints = True, then the force is limited to [-5, 5] and position is limited to [-1, 1]
    """
    def __init__(self, is_with_constraints = True, T = 150):
        ##### Dynamic Function ########
        n, m = 5, 1 # number of state = 4, number of action = 1, prediction horizon = 150
        h_constant = 0.02 # sampling time
        m_c = 1 # car mass
        m_p = 0.1 # pole mass
        l=0.5 # half pole length
        g = 9.8 # gravity
        x_u_var = sp.symbols('x_u:6') 
        theta_next = sp.atan2(x_u_var[2], x_u_var[3]) + h_constant * x_u_var[4]
        gamma = (x_u_var[5] + m_p*l*(x_u_var[4]**2)*x_u_var[2])/(m_c+m_p)
        dotdot_theta = (g*x_u_var[2]-x_u_var[3]*gamma)/(l*((4/3)-((m_p*(x_u_var[3]**2))/(m_c+m_p))))
        dotdot_x = gamma - (m_p*l*dotdot_theta*x_u_var[3])/(m_c+m_p)
        dynamic_function = sp.Array([  
            x_u_var[0] + h_constant*x_u_var[1],
            x_u_var[1] + h_constant*dotdot_x,
            sp.sin(theta_next),
            sp.cos(theta_next),
            x_u_var[4] + h_constant*dotdot_theta
        ])
        init_state = np.asarray([0, 0, 0.001, -1, 0],dtype=np.float64).reshape(-1,1)
        init_action = np.zeros((T, m, 1))
        if is_with_constraints: 
            constr = np.asarray([[-1, 1],           [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-10, 10]]) 
        else:
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
        ##### Objective Function ########
        C_matrix_diag = sp.symbols("c:6")
        r_vector = np.asarray([0, 0, 0, 1, 0, 0])
        add_param_obj = np.zeros((T, 6), dtype = np.float64)
        for tau in range(T):
            if tau < T-1:
                add_param_obj[tau] = np.asarray((0.1, 0.1, 1, 1, 0.1, 0.1))
            else: 
                add_param_obj[tau] = np.asarray((0, 0, 10000, 10000, 1000, 0))
        obj_fun = (x_u_var- r_vector)@np.diag(np.asarray(C_matrix_diag))@(x_u_var- r_vector)
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            constr = constr, 
                            init_state = init_state, 
                            init_action = init_action, 
                            obj_fun = obj_fun,
                            add_param_var = C_matrix_diag, 
                            add_param = add_param_obj)

    def play(self, logger_folder=None, no_iter = -1):
        """ If logger_folder exists and the result file is saved, then the specific iteration can be chosen to play the animation. \\

            Parameter
            ----------
            logger_folder : string
                The name of the logger folder
            no_iter : int
                The number of iteration to play the animation
        """
        fig, ax = super().create_plot(figsize=(5, 2), xlim=(-5, 5), ylim=(-1,1))
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        cart = patches.FancyBboxPatch((0, -0.1), 0.4, 0.2, "round,pad=0.02")
        cart.set_color('C0')
        pole = patches.FancyBboxPatch((0, 0), 0.04, 0.5, "round,pad=0.02")
        pole.set_color('C1')
        ax.add_patch(cart)
        ax.add_patch(pole)
        self._is_interrupted=False
        for i in range(self.T):
            angle = np.arctan2(trajectory[i,2,0], trajectory[i,3,0])
            t_start = ax.transData
            x = trajectory[i,0,0]-0.02*np.cos(angle)
            y = 0.02*np.sin(angle)
            rotate_center = t_start.transform([x, y])
            pole.set_x(x)
            pole.set_y(y)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -angle)
            t_end = t_start + t
            pole.set_transform(t_end)
            cart.set_x(trajectory[i,0]-0.2)
            fig.canvas.draw()
            plt.pause(0.01)
            if self._is_interrupted:
                return
        self._is_interrupted = True
