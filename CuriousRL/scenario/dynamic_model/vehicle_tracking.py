import numpy as np
import sympy as sp
from .dynamic_model import DynamicModel
from CuriousRL.utils.Logger import logger
from CuriousRL.data import ActionSpace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class VehicleTracking(DynamicModel):
    """In this example, a vehicle with 4 states and 2 actions, parks at (0, 0) heading to the top.
    We hope that the vechile finally can track the reference :math:`x=-10` with velocity 8 m/s, and heads to the right.
    The states and actions are listed as follows:

    - xu0(state): X position, limited to [-1, inf]
    - xu1(state): Y position
    - xu2(state): Heading angle
    - xu3(state): Velocity
    - xu4(action): Steering angle, limited to [-0.6, 0.6]
    - xu5(action): Acceleration, limited to [-3, 3]

    The dynamic model used in the example is introduced in ``CarParking``.

    :param is_with_constraints: Whether the box constraints of
        state and action variables are considered, defaults to True
    :type is_with_constraints: bool, optional
    :param T: Time horizon, defaults to 150
    :type T: int, optional
    """
    def __init__(self, is_with_constraints = True, T = 150):
        self._action_space = ActionSpace(action_range=[[-0.6, 0.6], [-3, 3]], action_type=["Continuous","Continuous"], action_info=["Steering Angle","Acceleration"])
        ##### Dynamic Function ########
        h_constant = 0.1 # sampling time
        xu_var = sp.symbols('x_u:6')
        d_constant = 3
        h_d_constanT = h_constant/d_constant
        b_function = d_constant \
                    + h_constant*xu_var[3]*sp.cos(xu_var[4]) \
                    -sp.sqrt(d_constant**2 - (h_constant**2)*(xu_var[3]**2)*(sp.sin(xu_var[4])**2))
        dynamic_function = sp.Array([  
                    xu_var[0] + b_function*sp.cos(xu_var[2]), 
                    xu_var[1] + b_function*sp.sin(xu_var[2]), 
                    xu_var[2] + sp.asin(h_d_constanT*xu_var[3]*sp.sin(xu_var[4])), 
                    xu_var[3]+h_constant*xu_var[5]
                ])
        init_state = np.asarray([0,0,np.pi/2,0],dtype=np.float64).reshape(-1,1)
        if is_with_constraints: 
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-0.6, 0.6], [-3, 3]]) 
        else:
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]) 
        ##### Objective Function ########
        C_matrix = np.diag([0.,1.,1.,1.,10.,10.])
        r_vector = np.asarray([0.,-10.,0.,8.,0.,0.])
        obj_fun = (xu_var - r_vector)@C_matrix@(xu_var - r_vector)
        super().__init__(   dynamic_function=dynamic_function, 
                            xu_var = xu_var, 
                            constr = constr, 
                            init_state = init_state, 
                            T=T,
                            obj_fun = obj_fun)

    def render(self):
        """ This method will render an image for the current state."""
        if self._fig is None:
            super()._create_plot(figsize=(8, 2), xlim=(-5,75), ylim=(-15,5))
            self._render_car = patches.FancyBboxPatch((0, 0), 3, 2, "round,pad=0.2")
            self._render_car.set_color('C0')
            self._ax.add_patch(self._render_car)
        angle = self._current_state[2]
        t_start = self._ax.transData
        x = self._current_state[0] + 1*np.sin(angle)
        y = self._current_state[1] - 1*np.cos(angle)
        rotate_center = t_start.transform([x, y])
        self._render_car.set_x(x)
        self._render_car.set_y(y)
        t = mpl.transforms.Affine2D().rotate_around(
            rotate_center[0], rotate_center[1], angle)
        t_end = t_start + t
        self._render_car.set_transform(t_end)
        self._fig.canvas.draw()
        plt.pause(0.001)

    @property
    def action_space(self):
        return self._action_space
