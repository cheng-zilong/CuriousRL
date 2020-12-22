import numpy as np
import sympy as sp
from .dynamic_model import DynamicModel
from CuriousRL.utils.Logger import logger
from CuriousRL.data import ActionSpace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class RoboticArmTracking(DynamicModel):
    """In this example, a robotic arm system with full dynamics is utilized to realize the tracking objective.
    The states and actions are listed as follows:

    - xu0(state): Angle of the first joint :math:`\\theta_1`
    - xu1(state): Anglular velocity of the first joint :math:`\\dot\\theta_1`
    - xu2(state): Angle of the second joint :math:`\\theta_2`
    - xu3(state): Anglular velocity of the second joint :math:`\\dot\\theta_2`
    - xu4(state): End effector X postition :math:`x`
    - xu5(state): End effector Y postition :math:`y`
    - xu6(action): Moment applied to the first joint :math:`\\tau_1`, limited to [-500, 500]
    - xu7(action): Moment applied to the second joint :math:`\\tau_2`, limited to [-500, 500]

    The dynamic model used in the example is introduced as follows.

    .. math::
        \\begin{bmatrix}\\ddot \\theta_1 \\\\ \\ddot \\theta_2 \\end{bmatrix}=H^{-1}(\\theta_1,\\theta_2) 
        Q(\\theta_1,\\theta_2, \\dot \\theta_1, \\dot \\theta_2)+H^{-1}(\\theta_1,\\theta_2) \\tau\\\\

    .. math::
        \\begin{array}{l}
        H(\\theta_1,\\theta_2)=\\left[\\begin{array}{cc}
        \\frac{1}{3} m_{1} l_{1}^{2}+m_{2} l_{1}^{2} & \\frac{1}{2} m_{2} l_{1} l_{2} \\cos \\left(\\theta_{1}-\\theta_{2}\\right) \\\\
        \\frac{1}{2} m_{2} l_{1} l_{2} \\cos \\left(\\theta_{1}-\\theta_{2}\\right) & \\frac{1}{3} m_{2} l_{2}^{2}
        \\end{array}\\right]
        \\end{array}

    .. math::
        \\begin{array}{rCl}
        Q(\\theta_1,\\theta_2, \\dot \\theta_1, \\dot \\theta_2)&=&\\begin{bmatrix}
        Q_1\\\\
        Q_2
        \\end{bmatrix}(\\theta_1,\\theta_2, \\dot \\theta_1, \\dot \\theta_2) \\\\
        \\tau&=&\\begin{bmatrix}
        \\tau_{1} \\\\
        \\tau_{2}
        \\end{bmatrix} \\\\
        \\end{array}

    .. math::
        \\begin{array}{rCl}
        Q_1&=&-\\frac{1}{2} m_{2} l_{1} l_{2} \\dot{\\theta}_{2}^{2} \\sin \\left(\\theta_{1}-\\theta_{2}\\right)+\\frac{1}{2} m_{1} g l_{1} \\sin \\left(\\theta_{1}\\right) 
        +m_{2} g l_{1} \\sin \\left(\\theta_{1}\\right)\\\\
        Q_{2}&=&\\frac{1}{2} m_{2} l_{1} l_{2} \\dot{\\theta}_{1}^{2} \\sin \\left(\\theta_{1}-\\theta_{2}\\right)+\\frac{1}{2} m_{2} g l_{2} \\sin \\left(\\theta_{2}\\right)
        \\end{array}



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


    def __init__(self, is_with_constraints = True, T = 100, x = 2, y = 2):
        self._action_space = ActionSpace(action_range=[[-500, 500], [-500, 500]], action_type=["Continuous", "Continuous"], action_info=["Moment 1", "Moment 2"])
        ##### Dynamic Function ########
        xu_var = sp.symbols('x_u:8')
        m1 = 1
        m2 = 2
        self.l1 = 1
        self.l2 = 2
        g = 9.8
        h = 0.01 # sampling time
        H = sp.Matrix([
            [((1/3)*m1 + m2)*(self.l1**2),          (1/2)*m2*self.l1*self.l2*sp.cos(xu_var[0] - xu_var[2])],
            [(1/2)*m2*self.l1*self.l2*sp.cos(xu_var[0] - xu_var[2]),               (1/3)*m2*(self.l2**2)   ]
        ])
        H_inv = H.inv()
        Q = np.asarray([
            [-0.5*m2*self.l1*self.l2*(xu_var[3]**2)*sp.sin(xu_var[0] - xu_var[2]) + 0.5*m1*g*self.l1*sp.sin(xu_var[0])+m2*g*self.l1*sp.sin(xu_var[0])],
            [0.5*m2*self.l1*self.l2*(xu_var[1]**2)*sp.sin(xu_var[0] - xu_var[2]) + 0.5*m2*g*self.l2*sp.sin(xu_var[2])]
            ])
        tau =  np.asarray([
            [xu_var[6]],
            [xu_var[7]]
            ])
        temp = H_inv@Q + H_inv@tau
        theta1_ddot = temp[0,0]
        theta2_ddot = temp[1,0]
        dynamic_function = sp.Array([  
            xu_var[0] + h*xu_var[1],
            xu_var[1] + h*theta1_ddot,
            xu_var[2] + h*xu_var[3],
            xu_var[3] + h*theta2_ddot,
            self.l1*sp.sin(xu_var[0]) + self.l2*sp.sin(xu_var[2]),
            self.l1*sp.cos(xu_var[0]) + self.l2*sp.cos(xu_var[2])])
        init_state = np.asarray([0, 0, 0, 0, 0, self.l1+self.l2],dtype=np.float64).reshape(-1,1)
        if is_with_constraints: 
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-500, 500], [-500, 500]]) 
        else:
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
        ##### Objective Function ########
        position_var = sp.symbols("p:2") # x and y
        C_matrix =    np.diag([0.,      10.,     0.,        10.,          10000,                             10000,                 0.01,           0.01])
        r_vector = np.asarray([0.,       0.,     0.,         0.,          position_var[0],            position_var[1],              0.,              0.])
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
            super()._create_plot(xlim=(-4,4), ylim=(-4,4))
            self._render_pole1 = patches.FancyBboxPatch((0, 0), 0.04, self.l1, "round,pad=0.02")
            self._render_pole1.set_color('C0')
            self._render_pole2 = patches.FancyBboxPatch((0, 0), 0.04, self.l2, "round,pad=0.02")
            self._render_pole2.set_color('C1')
            self._ax.add_patch(self._render_pole1)
            self._ax.add_patch(self._render_pole2)
        t_start = self._ax.transData
        # draw pole1
        x1 = -0.02*np.cos(self._current_state[0])
        y1 =  0.02*np.sin(self._current_state[0])
        rotate_center = t_start.transform([x1, y1])
        self._render_pole1.set_x(x1)
        self._render_pole1.set_y(y1)
        t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self._current_state[0])
        t_end = t_start + t
        self._render_pole1.set_transform(t_end)
        # draw pole2
        x2 = self.l1*np.sin(self._current_state[0]) - 0.02*np.cos(self._current_state[2])
        y2 = self.l1*np.cos(self._current_state[0]) + 0.02*np.sin(self._current_state[2])
        rotate_center = t_start.transform([x2, y2])
        self._render_pole2.set_x(x2)
        self._render_pole2.set_y(y2)
        t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self._current_state[2])
        t_end = t_start + t
        self._render_pole2.set_transform(t_end)
        self._fig.canvas.draw()
        plt.pause(0.001)

    @property
    def action_space(self):
        """ The ``ActionSpace`` of the scenario."""
        return self._action_space
