#%%
import numpy as np
import sympy as sp
from .dynamic_model import DynamicModel
from CuriousRL.utils.Logger import logger
from CuriousRL.data import ActionSpace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class CartPoleSwingUp2(DynamicModel):
    """In this example, a cartpole system is stable with the pole lying downsode. The cart is static at 
    the original point. The states and actions are listed as follows:

    - xu0(state): Cart position, limited to [-1, 1]
    - xu1(state): Cart Velocity
    - xu2(state): Sin(pole angle)
    - xu3(state): Cos(pole angle)
    - xu4(state): Pole angular velocity
    - xu5(action): Force applied to the cart, limited to [-10, 10]

    The dynamic model used in the example is introduced in ``CartPoleSwingUp1``.

    :param is_with_constraints: Whether the box constraints of
        state and action variables are considered, defaults to True
    :type is_with_constraints: bool, optional
    :param T: Time horizon, defaults to 150
    :type T: int, optional
    """

    def __init__(self, is_with_constraints = True, T = 150):
        self._action_space = ActionSpace(action_range=[[-10, 10]], action_type=["Continuous"], action_info=["Force"])
        ##### Dynamic Function ########
        h_constant = 0.02 # sampling time
        m_c = 1 # car mass
        m_p = 0.1 # pole mass
        l=0.5 # half pole length
        g = 9.8 # gravity
        xu_var = sp.symbols('x_u:6') 
        theta_next = sp.atan2(xu_var[2], xu_var[3]) + h_constant * xu_var[4]
        gamma = (xu_var[5] + m_p*l*(xu_var[4]**2)*xu_var[2])/(m_c+m_p)
        dotdot_theta = (g*xu_var[2]-xu_var[3]*gamma)/(l*((4/3)-((m_p*(xu_var[3]**2))/(m_c+m_p))))
        dotdot_x = gamma - (m_p*l*dotdot_theta*xu_var[3])/(m_c+m_p)
        dynamic_function = sp.Array([  
            xu_var[0] + h_constant*xu_var[1],
            xu_var[1] + h_constant*dotdot_x,
            sp.sin(theta_next),
            sp.cos(theta_next),
            xu_var[4] + h_constant*dotdot_theta
        ])
        init_state = np.asarray([0, 0, 0.001, -1, 0],dtype=np.float64).reshape(-1,1)
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
        obj_fun = (xu_var- r_vector)@np.diag(np.asarray(C_matrix_diag))@(xu_var- r_vector)
        super().__init__(   dynamic_function=dynamic_function, 
                            xu_var = xu_var, 
                            constr = constr, 
                            init_state = init_state, 
                            T=T,
                            obj_fun = obj_fun,
                            add_param_var = C_matrix_diag, 
                            add_param = add_param_obj)

    def render(self):
        """ This method will render an image for the current state."""
        if self._fig is None:
            super()._create_plot(figsize=(5, 2), xlim=(-5, 5), ylim=(-1,1))
            self._render_cart = patches.FancyBboxPatch((0, -0.1), 0.4, 0.2, "round,pad=0.02")
            self._render_cart.set_color('C0')
            self._render_pole = patches.FancyBboxPatch((0, 0), 0.04, 0.5, "round,pad=0.02")
            self._render_pole.set_color('C1')
            self._ax.add_patch(self._render_cart)
            self._ax.add_patch(self._render_pole)

        angle = np.arctan2(self._current_state[2], self._current_state[3])
        t_start = self._ax.transData
        x = self._current_state[0]-0.02*np.cos(angle)
        y = 0.02*np.sin(angle)
        rotate_center = t_start.transform([x, y])
        self._render_pole.set_x(x)
        self._render_pole.set_y(y)
        t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -angle)
        t_end = t_start + t
        self._render_pole.set_transform(t_end)
        self._render_cart.set_x(self._current_state[0]-0.2)
        self._fig.canvas.draw()
        plt.pause(0.001)

    @property
    def action_space(self):
        """ The ``ActionSpace`` of the scenario."""
        return self._action_space
