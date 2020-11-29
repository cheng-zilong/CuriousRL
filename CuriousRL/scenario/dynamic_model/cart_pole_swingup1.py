#%%
import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelWrapper
from CuriousRL.utils.Logger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class CartPoleSwingUp1(DynamicModelWrapper):
    """ In this example, the cartpole system is static at 0, 0, heading to the postive direction of the y axis\\
        We hope the vechile can tracking the reference y=-10 with the velocity 8, and head to the right\\
        x0: angle, x1: angular velocity, x2: position, x3: velocity, x4: force\\
        If is_with_constraints = True, then the force is limited to [-5, 5] and position is limited to [-1, 1]
    """
    def __init__(self, is_with_constraints = True, T = 150):
        ##### Dynamic Function ########
        n, m = 4, 1 # number of state = 4, number of input = 1, prediction horizon = 150
        h_constant = 0.02 # sampling time
        m_c = 1 # car mass
        m_p = 0.1 # pole mass
        l=0.5 # half pole length
        h = 0.02 # sampling time
        g = 9.8 # gravity
        x_u_var = sp.symbols('x_u:5')
        gamma = (x_u_var[4] + m_p*l*(x_u_var[1]**2)*sp.sin(x_u_var[0]))/(m_c+m_p)
        dotdot_theta = (g*sp.sin(x_u_var[0])-sp.cos(x_u_var[0])*gamma)/(l*((4/3)-((m_p*(sp.cos(x_u_var[0])**2))/(m_c+m_p))))
        dotdot_x = gamma - (m_p*l*dotdot_theta*sp.cos(x_u_var[0]))/(m_c+m_p)
        dynamic_function = sp.Array([  
            x_u_var[0] + h*x_u_var[1],
            x_u_var[1] + h*dotdot_theta,
            x_u_var[2] + h*x_u_var[3],
            x_u_var[3] + h*dotdot_x])
        init_state = np.asarray([np.pi, 0, 0, 0],dtype=np.float64).reshape(-1,1)
        init_input = np.zeros((T, m, 1))
        if is_with_constraints: 
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-1, 1], [-5, 5]]) 
        else:
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
        ##### Objective Function ########
        C_matrix_diag = sp.symbols("c:5")
        add_param_obj = np.zeros((T, 5), dtype = np.float64)
        for tau in range(T):
            if tau < T-1:
                add_param_obj[tau] = np.asarray((1, 0.1, 1, 1, 1))
            else: 
                add_param_obj[tau] = np.asarray((10000, 1000, 0, 0, 0)) # terminal objective function
        obj_fun = x_u_var@np.diag(np.asarray(C_matrix_diag))@x_u_var
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            constr = constr, 
                            init_state = init_state, 
                            init_input = init_input, 
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
            t_start = ax.transData
            x = trajectory[i,2,0]-0.02*np.cos(trajectory[i,0,0])
            y = 0.02*np.sin(trajectory[i,0,0])
            rotate_center = t_start.transform([x, y])
            pole.set_x(x)
            pole.set_y(y)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -trajectory[i,0,0])
            t_end = t_start + t
            pole.set_transform(t_end)
            cart.set_x(trajectory[i,2]-0.2)
            fig.canvas.draw()
            plt.pause(0.01)
            if self._is_interrupted:
                return
        self._is_interrupted = True
# %%

