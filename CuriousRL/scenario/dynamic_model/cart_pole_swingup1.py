#%%
import numpy as np
import sympy as sp
from .dynamic_model import DynamicModel
from CuriousRL.utils.Logger import logger
from CuriousRL.data import ActionSpace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class CartPoleSwingUp1(DynamicModel):
    """In this example, a cartpole system is stable with the pole lying downsode. The cart is static at 
    the original point. The states and actions are listed as follows:

    - xu0(state): Pole angle
    - xu1(state): Pole angular velocity
    - xu2(state): Cart position, limited to [-1, 1]
    - xu3(state): Cart Velocity
    - xu4(action): Force applied to the cart, limited to [-10, 10]

    The dynamic model used in the example is introduced as follows.
    
    The state vector of the cartpole dynamic function is defined as 
    :math:`x=[\\theta \\quad \\omega \\quad p \\quad v]^T`,
    where :math:`\\theta` and :\math:`\\omega` denote the angle between the pole and the
    vertical direction and its corresponding angular velocity, respectively; 
    :math:`p` and :math:`v` denote the position and the velocity of the cart, respectively. 
    The action variable of the cartpole system is denoted by :math:`F`, 
    which means the force applied to the cart. 
    The gravitational acceleration, sampling time, mass of the cart, mass of the pole, 
    and half length of the pole are denoted by :math:`g,h,m_c,m_p,\\ell`, respectively. 
    Further define the angular acceleration as :math:`\\alpha` and the acceleration of the cart 
    as :math:`a`.

    .. math::
        \\begin{array}{rCl}
            \\alpha(\\theta,\\omega,F)&=& \\dfrac{g \sin (\\theta)+\\cos (\\theta)\\left(\\frac{-F-m_{p} \\ell \\omega^{2} \\sin (\\theta)}{m_{c}+m_{p}}\\right)}{\\ell\left(\\frac{4}{3}-\\frac{m_{p} \\cos ^{2} (\\theta)}{m_{c}+m_{p}}\\right)}\\nonumber\\\\
            a(\\theta,\\omega,F)&=&\\dfrac{F+m_{p} \\ell\\left(\\omega^{2} \\sin (\\theta)-\\alpha \\cos (\\theta)\\right)}{m_{c}+m_{p}}.
        \end{array}


    It follows that the cartpole model can be represented by

    .. math::
        \\begin{array}{rCl}
            \\theta(\\tau+1) &=& \\theta(\\tau) + h\\omega(\\tau)\\\\
            \\omega(\\tau+1) &=& \\omega(\\tau) + h\\alpha\Big(\\theta(\\tau),\omega(\\tau),F(\\tau)\Big)\\\\
            p(\\tau+1) &=& p(\\tau) + hv(\\tau)\\\\
            v(\\tau+1) &=& v(\\tau) + ha\\Big(\\theta(\\tau),\\omega(\\tau),F(\\tau)\\Big).
        \end{array}

    :param is_with_constraints: Whether the box constraints of
        state and action variables are considered, defaults to True
    :type is_with_constraints: bool, optional
    :param T: Time horizon, defaults to 150
    :type T: int, optional
    """
    def __init__(self, is_with_constraints = True, T = 150):
        self._action_space = ActionSpace(action_range=[[-10, 10]], action_type=["Continuous"], action_info=["Force"])
        ##### Dynamic Function ########
        m_c = 1 # car mass
        m_p = 0.1 # pole mass
        l=0.5 # half pole length
        h = 0.02 # sampling time
        g = 9.8 # gravity
        xu_var = sp.symbols('x_u:5')
        gamma = (xu_var[4] + m_p*l*(xu_var[1]**2)*sp.sin(xu_var[0]))/(m_c+m_p)
        dotdot_theta = (g*sp.sin(xu_var[0])-sp.cos(xu_var[0])*gamma)/(l*((4/3)-((m_p*(sp.cos(xu_var[0])**2))/(m_c+m_p))))
        dotdot_x = gamma - (m_p*l*dotdot_theta*sp.cos(xu_var[0]))/(m_c+m_p)
        dynamic_function = sp.Array([  
            xu_var[0] + h*xu_var[1],
            xu_var[1] + h*dotdot_theta,
            xu_var[2] + h*xu_var[3],
            xu_var[3] + h*dotdot_x])
        init_state = np.asarray([np.pi, 0, 0, 0],dtype=np.float64).reshape(-1,1)
        if is_with_constraints: 
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-1, 1],  [-np.inf, np.inf], [-10, 10]]) 
        else:
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
        ##### Objective Function ########
        C_matrix_diag = sp.symbols("c:5")
        add_param_obj = np.zeros((T, 5), dtype = np.float64)
        for tau in range(T):
            if tau < T-1:
                add_param_obj[tau] = np.asarray((1, 0.1, 1, 1, 0.1))
            else: 
                add_param_obj[tau] = np.asarray((10000, 1000, 0, 0, 0)) # terminal objective function
        obj_fun = xu_var@np.diag(np.asarray(C_matrix_diag))@xu_var
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
        t_start = self._ax.transData
        x = self._current_state[2]-0.02*np.cos(self._current_state[0])
        y = 0.02*np.sin(self._current_state[0])
        rotate_center = t_start.transform([x, y])
        self._render_pole.set_x(x)
        self._render_pole.set_y(y)
        t = mpl.transforms.Affine2D().rotate_around(rotate_center[0], rotate_center[1], -self._current_state[0])
        t_end = t_start + t
        self._render_pole.set_transform(t_end)
        self._render_cart.set_x(self._current_state[2]-0.2)
        self._fig.canvas.draw()
        plt.pause(0.001)
        
    @property
    def action_space(self):
        """ The ``ActionSpace`` of the scenario."""
        return self._action_space

    @property
    def state_shape(self):
        return (4,)

