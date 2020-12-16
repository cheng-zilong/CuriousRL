import numpy as np
import sympy as sp
from .dynamic_model import DynamicModel
from CuriousRL.utils.Logger import logger
from CuriousRL.data import ActionSpace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class ThreeLinkPlanarManipulator(DynamicModel):
    """In this example, a simple two link planar manipulator with simple dynamics 
    is utilized to realize the tracking objective.
    The states and actions are listed as follows:

    - xu0(state): Angle of the first joint :math:`\\theta_1`
    - xu1(state): Anglular velocity of the first joint :math:`\\dot\\theta_1`
    - xu2(state): Angle of the second joint :math:`\\theta_2`
    - xu3(state): Anglular velocity of the second joint :math:`\\dot\\theta_2`
    - xu4(state): Angle of the third joint :math:`\\theta_3`
    - xu5(state): Anglular velocity of the third joint :math:`\\dot\\theta_3`
    - xu6(state): End effector X postition :math:`x`
    - xu7(state): End effector Y postition :math:`y`
    - xu8(action): Moment applied to the first joint :math:`\\tau_1`, limited to [-100, 100]
    - xu9(action): Moment applied to the second joint :math:`\\tau_2`, limited to [-100, 100]
    - xu10(action): Moment applied to the third joint :math:`\\tau_3`, limited to [-100, 100]

    .. math::
        \\begin{bmatrix}
        \\theta_1(k+1)\\\\
        \\dot \\theta_1(k+1)\\\\
        \\theta_2(k+1)\\\\
        \\dot \\theta_2(k+1)\\\\
        \\theta_3(k+1)\\\\
        \\dot \\theta_3(k+1)\\\\
        x(k+1)\\\\
        y(k+1)
        \\end{bmatrix}=
        \\begin{bmatrix}
        \\theta_1(k)+\\dot \\theta_1(k)\\\\
        \\dot \\theta_1(k)+\\tau_1(k)\\\\
        \\theta_2(k)+\\dot \\theta_2(k)\\\\
        \\dot \\theta_2(k)+\\tau_2(k)\\\\
        \\theta_3(k)+\\dot \\theta_3(k)\\\\
        \\dot \\theta_3(k)+\\tau_3(k)\\\\
        \\ell_1\\sin\\big(\\theta_1(k)\\big)+\\ell_2\\sin\\big(\\theta_1(k)+\\theta_2(k)\\big)+\\ell_3\\sin\\big(\\theta_1(k)+\\theta_2(k)+\\theta_3(k)\\big)\\\\
        \\ell_1\\cos\\big(\\theta_1(k)\\big)+\\ell_2\\cos\\big(\\theta_1(k)+\\theta_2(k)\\big)+\\ell_3\\cos\\big(\\theta_1(k)+\\theta_2(k)+\\theta_3(k)\\big)\\\\
        \\end{bmatrix}

    :param is_with_constraints: Whether the box constraints of
        state and action variables are considered, defaults to True
    :type is_with_constraints: bool, optional
    :param T: Time horizon, defaults to 100
    :type T: int, optional
    :param x: Target position X, defaults to 2
    :type x: int, optional
    :param y: Target position Y, defaults to 2
    :type y: int, optional
    """
    def __init__(self, T = 100, x = 2, y = 2):
        self._action_space = ActionSpace(action_range=[[-100, 100], [-100, 100], [-100, 100]], action_type=["Continuous", "Continuous", "Continuous"], action_info=["Moment 1", "Moment 2", "Moment 3"])
        ##### Dynamic Function ########
        xu_var = sp.symbols('x_u:11')
        h = 0.01 # sampling time
        self.l1 = 1
        self.l2 = 2
        self.l3 = 2
        dynamic_function = sp.Array([  
            xu_var[0] + h*xu_var[1],
            xu_var[1] + h*xu_var[8],
            xu_var[2] + h*xu_var[3],
            xu_var[3] + h*xu_var[9],
            xu_var[4] + h*xu_var[5],
            xu_var[5] + h*xu_var[10],
            self.l1*sp.sin(xu_var[0]) + self.l2*sp.sin(xu_var[0]+xu_var[2]) + self.l3*sp.sin(xu_var[0]+xu_var[2]+xu_var[4]),
            self.l1*sp.cos(xu_var[0]) + self.l2*sp.cos(xu_var[0]+xu_var[2]) + self.l3*sp.cos(xu_var[0]+xu_var[2]+xu_var[4])])
        init_state = np.asarray([0, 0, 0, 0, 0, 0, 0, self.l1+self.l2+self.l3], dtype=np.float64).reshape(-1,1)
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
        obj_fun = (xu_var - r_vector)@C_matrix@(xu_var - r_vector) 
        add_param = np.hstack([x*np.ones((T, 1)), y*np.ones((T, 1))])
        super().__init__(   dynamic_function=dynamic_function, 
                            xu_var = xu_var, 
                            constr = constr, 
                            init_state = init_state, 
                            T=T,
                            obj_fun = obj_fun,
                            add_param_var= position_var,
                            add_param= add_param)

    def render(self):
        """ This method will render an image for the current state."""
        if self._fig is None:
            super()._create_plot(xlim=(-6,6), ylim=(-6,6))
            self._render_pole1 = patches.FancyBboxPatch((0, 0), 0.04, self.l1, "round,pad=0.02")
            self._render_pole1.set_color('C0')
            self._render_pole2 = patches.FancyBboxPatch((0, 0), 0.04, self.l2, "round,pad=0.02")
            self._render_pole2.set_color('C1')
            self._render_pole3 = patches.FancyBboxPatch((0, 0), 0.04, self.l3, "round,pad=0.02")
            self._render_pole3.set_color('C2')
            self._ax.add_patch(self._render_pole1)
            self._ax.add_patch(self._render_pole2)
            self._ax.add_patch(self._render_pole3)
        # draw pole1
        t_start = self._ax.transData
        x1 = -0.02*np.cos(self._current_state[0])
        y1 =  0.02*np.sin(self._current_state[0])
        rotate_center = t_start.transform([x1, y1])
        self._render_pole1.set_x(x1)
        self._render_pole1.set_y(y1)
        t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self._current_state[0])
        t_end = t_start + t
        self._render_pole1.set_transform(t_end)
        # draw pole2
        x2 = self.l1*np.sin(self._current_state[0]) - 0.02*np.cos(self._current_state[0]+self._current_state[2])
        y2 = self.l1*np.cos(self._current_state[0]) + 0.02*np.sin(self._current_state[0]+self._current_state[2])
        rotate_center = t_start.transform([x2, y2])
        self._render_pole2.set_x(x2)
        self._render_pole2.set_y(y2)
        t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self._current_state[0]-self._current_state[2])
        t_end = t_start + t
        self._render_pole2.set_transform(t_end)
        # draw pole3
        x3 = self.l1*np.sin(self._current_state[0]) + self.l2*np.sin(self._current_state[0]+self._current_state[2]) - 0.02*np.cos(self._current_state[0]+self._current_state[2]+self._current_state[4])
        y3 = self.l1*np.cos(self._current_state[0]) + self.l2*np.cos(self._current_state[0]+self._current_state[2]) + 0.02*np.sin(self._current_state[0]+self._current_state[2]+self._current_state[4])
        rotate_center = t_start.transform([x3, y3])
        self._render_pole3.set_x(x3)
        self._render_pole3.set_y(y3)
        t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self._current_state[0]-self._current_state[2]-self._current_state[4])
        t_end = t_start + t
        self._render_pole3.set_transform(t_end)
        self._fig.canvas.draw()
        plt.pause(0.001)

    @property
    def action_space(self):
        """ The ``ActionSpace`` of the scenario."""
        return self._action_space