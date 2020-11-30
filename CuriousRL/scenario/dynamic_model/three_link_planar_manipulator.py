import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelWrapper
from CuriousRL.utils.Logger import logger
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class ThreeLinkPlanarManipulator(DynamicModelWrapper):
    """ In this example, the vehicle packing at 0, 0, heading to the top\\
        We hope the vechile can pack at 0, 0, and head to the right\\
        x0: position_x, x1: position_y, x2: heading anglue, x3: velocity, x4: steering angle, x5: acceleration\\
        If is_with_constraints = True, then the steering angle is limited to [-0.5, 0.5], acceleration is limited to [-2, 2]
    """
    def __init__(self, T = 100, x = 2, y = 2):
        ##### Dynamic Function ########
        # x0: theta1 
        # x1: theta1 dot 
        # x2: theta2
        # x3: theta2 dot
        # x4: theta3
        # x5: theta3 dot
        # x6: teminal x
        # x7: teminal y
        # x8: tau1
        # x9: tau2
        # x10: tau3
        
        n, m = 8, 3 # number of state = 6, number of action = 2, prediction horizon = 500
        x_u_var = sp.symbols('x_u:11')
        h = 0.01 # sampling time
        self.l1 = 1
        self.l2 = 2
        self.l3 = 2
        dynamic_function = sp.Array([  
            x_u_var[0] + h*x_u_var[1],
            x_u_var[1] + h*x_u_var[8],
            x_u_var[2] + h*x_u_var[3],
            x_u_var[3] + h*x_u_var[9],
            x_u_var[4] + h*x_u_var[5],
            x_u_var[5] + h*x_u_var[10],
            self.l1*sp.sin(x_u_var[0]) + self.l2*sp.sin(x_u_var[0]+x_u_var[2]) + self.l3*sp.sin(x_u_var[0]+x_u_var[2]+x_u_var[4]),
            self.l1*sp.cos(x_u_var[0]) + self.l2*sp.cos(x_u_var[0]+x_u_var[2]) + self.l3*sp.cos(x_u_var[0]+x_u_var[2]+x_u_var[4])])
        init_state = np.asarray([0, 0, 0, 0, 0, 0, 0, self.l1+self.l2+self.l3], dtype=np.float64).reshape(-1,1)
        init_action = np.zeros((T,m,1))
        constr = np.asarray([   [-np.inf, np.inf], 
                                [-np.inf, np.inf], 
                                [-np.inf, np.inf], 
                                [-np.inf, np.inf], 
                                [-np.inf, np.inf], 
                                [-np.inf, np.inf], 
                                [-np.inf, np.inf],
                                [-np.inf, np.inf], 
                                [-100, 100], [-100, 100], [-100, 100]]) 
        ##### Objective Function ########
        position_var = sp.symbols("p:2") # x and y
        C_matrix =    np.diag([0.,      10.,     0.,        10.,          0.,        10.,         10000.,                                10000,         1,           1,          1])
        r_vector = np.asarray([0.,       0.,     0.,         0.,          0.,         0.,        position_var[0],            position_var[1],           0.,          0.,         0.])
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
        fig, ax = super().create_plot(xlim=(-6,6), ylim=(-6,6))
        trajectory = np.asarray(logger.read_from_json(logger_folder, no_iter)["trajectory"])
        pole1 = patches.FancyBboxPatch((0, 0), 0.04, self.l1, "round,pad=0.02")
        pole1.set_color('C0')
        pole2 = patches.FancyBboxPatch((0, 0), 0.04, self.l2, "round,pad=0.02")
        pole2.set_color('C1')
        pole3 = patches.FancyBboxPatch((0, 0), 0.04, self.l3, "round,pad=0.02")
        pole3.set_color('C2')
        ax.add_patch(pole1)
        ax.add_patch(pole2)
        ax.add_patch(pole3)
        self._is_interrupted = False
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
            x2 = self.l1*np.sin(self.play_trajectory_current[0]) - 0.02*np.cos(self.play_trajectory_current[0]+self.play_trajectory_current[2])
            y2 = self.l1*np.cos(self.play_trajectory_current[0]) + 0.02*np.sin(self.play_trajectory_current[0]+self.play_trajectory_current[2])
            rotate_center = t_start.transform([x2, y2])
            pole2.set_x(x2)
            pole2.set_y(y2)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self.play_trajectory_current[0]-self.play_trajectory_current[2])
            t_end = t_start + t
            pole2.set_transform(t_end)
            # draw pole3
            x3 = self.l1*np.sin(self.play_trajectory_current[0]) + self.l2*np.sin(self.play_trajectory_current[0]+self.play_trajectory_current[2]) - 0.02*np.cos(self.play_trajectory_current[0]+self.play_trajectory_current[2]+self.play_trajectory_current[4])
            y3 = self.l1*np.cos(self.play_trajectory_current[0]) + self.l2*np.cos(self.play_trajectory_current[0]+self.play_trajectory_current[2]) + 0.02*np.sin(self.play_trajectory_current[0]+self.play_trajectory_current[2]+self.play_trajectory_current[4])
            rotate_center = t_start.transform([x3, y3])
            pole3.set_x(x3)
            pole3.set_y(y3)
            t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self.play_trajectory_current[0]-self.play_trajectory_current[2]-self.play_trajectory_current[4])
            t_end = t_start + t
            pole3.set_transform(t_end)
            fig.canvas.draw()
            plt.pause(0.01)
            if self._is_interrupted:
                return
        self._is_interrupted = True

