import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelWrapper
from CuriousRL.utils.Logger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from scipy.linalg import block_diag 

class SneakRobotTracking(DynamicModelWrapper):
    """ In this example, the vehicle packing at 0, 0, heading to the top\\
        We hope the vechile can pack at 0, 0, and head to the right\\
        x0: position_x, x1: position_y, x2: heading anglue, x3: velocity, x4: steering angle, x5: acceleration\\
        If is_with_constraints = True, then the steering angle is limited to [-0.5, 0.5], acceleration is limited to [-2, 2]
    """
    def __init__(self, is_with_constraints = True, T = 100, x = 2, y = 2):
        ##### Dynamic Function ########
        # x0-x3: theta
        # x4-x7: dot theta
        # x8-x10: phi
        # x11-x14: position_x
        # x15-x18: position_y
        # x19, x20, x21, x22: p_x, p_y, dot p_x, dot p_y
        # x23, x24, x25, tau0 .... tau2

        x_u_var = sp.symbols('x_u:26')
        J = 1.6e-3
        m = 1
        l = 0.07
        A = np.asarray([[1, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 1, 1]], dtype=np.float64)
        D = np.asarray([[1, -1, 0, 0],
                        [0, 1, -1, 0],
                        [0, 0, 1, -1]], dtype=np.float64)
        e = np.ones((4,1))
        E = block_diag(e,e)
        b = np.zeros((3,1))
        b[-1] = 1
        N = A.T@np.linalg.inv(D@D.T)@D
        V = A.T@np.linalg.inv(D@D.T)@A
        mu_t = 1
        mu_n = 3
        S_theta = np.diag([sp.sin(n) for n in x_u_var[0:4]])
        C_theta = np.diag([sp.cos(n) for n in x_u_var[0:4]])
        H = np.vstack([N.T@S_theta, -N.T@C_theta])
        Q_theta = -np.vstack([   np.hstack([mu_t*C_theta@C_theta + mu_n*S_theta@S_theta, (mu_t-mu_n)*S_theta@C_theta]),
                                np.hstack([(mu_t-mu_n)*S_theta@C_theta, mu_t*C_theta@C_theta + mu_n*S_theta@S_theta])])
        M = sp.Matrix(J*np.eye(4) + m*(l**2)*S_theta@V@S_theta + m*(l**2)*C_theta@V@C_theta)
        M_inv = sp.Inverse(M)
        W = m*(l**2)*C_theta@V@S_theta - m*(l**2)*S_theta@V@C_theta
        f_R = l*Q_theta@H@sp.Matrix(x_u_var[6:12])+Q_theta@E@sp.Matrix([x_u_var[31], x_u_var[32]])
        M_ddot_theta = W@sp.Matrix(x_u_var[6:12]) + l*H.T@f_R + D.T@sp.Matrix(x_u_var[33:38])


        theta1_ddot = temp[0,0]
        theta2_ddot = temp[1,0]
        dynamic_function = sp.Array([  
            x_u_var[0] + h*x_u_var[1],
            x_u_var[1] + h*theta1_ddot,
            x_u_var[2] + h*x_u_var[3],
            x_u_var[3] + h*theta2_ddot,
            self.l1*sp.sin(x_u_var[0]) + self.l2*sp.sin(x_u_var[2]),
            self.l1*sp.cos(x_u_var[0]) + self.l2*sp.cos(x_u_var[2])])
        init_state = np.asarray([0, 0, 0, 0, 0, self.l1+self.l2],dtype=np.float64).reshape(-1,1)
        init_action = np.zeros((T,m,1))
        if is_with_constraints: 
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-500, 500], [-500, 500]]) 
        else:
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
        ##### Objective Function ########
        position_var = sp.symbols("p:2") # x and y
        C_matrix =    np.diag([0.,      10.,     0.,        10.,          10000,                             10000,                 0.01,           0.01])
        r_vector = np.asarray([0.,       0.,     0.,         0.,          position_var[0],            position_var[1],              0.,              0.])
        obj_fun = (x_u_var - r_vector)@C_matrix@(x_u_var - r_vector) 
        add_param = np.hstack([x*np.ones((T, 1)), y*np.ones((T, 1))])
        super().__init__(   dynamic_function=dynamic_function, 
                            x_u_var = x_u_var, 
                            constr = constr, 
                            init_state = init_state, 
                            init_action = init_action, 
                            obj_fun = obj_fun,
                            add_param_var= position_var,
                            add_param= add_param)

    def play(self, logger_folder=None, no_iter = -1):
        """ If logger_folder exists and the result file is saved, then the specific iteration can be chosen to play the animation. \\

            Parameter
            ----------
            logger_folder : string
                The name of the logger folder
            no_iter : int
                The number of iteration to play the animation
        """
        fig, ax = super().create_plot(figsize=(4, 4), xlim=(-4,4), ylim=(-4,4))
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        pole1 = patches.FancyBboxPatch((0, 0), 0.04, self.l1, "round,pad=0.02")
        pole1.set_color('C0')
        pole2 = patches.FancyBboxPatch((0, 0), 0.04, self.l2, "round,pad=0.02")
        pole2.set_color('C1')
        ax.add_patch(pole1)
        ax.add_patch(pole2)
        self._is_interrupted=False
        for i in range(self.T):
            self.play_trajectory_current = trajectory[i,:,0]
            # draw pole1
            t_start = ax.transData
            x1 = -0.02*np.cos(self.play_trajectory_current[0])
            y1 =  0.02*np.sin(self.play_trajectory_current[0])
            rotate_center = t_start.transform([x1, y1])
            pole1.set_x(x1)
            pole1.set_y(y1)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self.play_trajectory_current[0])
            t_end = t_start + t
            pole1.set_transform(t_end)
            # draw pole2
            x2 = self.l1*np.sin(self.play_trajectory_current[0]) - 0.02*np.cos(self.play_trajectory_current[2])
            y2 = self.l1*np.cos(self.play_trajectory_current[0]) + 0.02*np.sin(self.play_trajectory_current[2])
            rotate_center = t_start.transform([x2, y2])
            pole2.set_x(x2)
            pole2.set_y(y2)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self.play_trajectory_current[2])
            t_end = t_start + t
            pole2.set_transform(t_end)
            fig.canvas.draw()
            plt.pause(0.001)
            if self._is_interrupted:
                return
        self._is_interrupted = True
