import numpy as np
import sympy as sp
from .dynamic_model import DynamicModelWrapper
from CuriousRL.utils.Logger import logger
from CuriousRL.data import ActionSpace
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class CarParking(DynamicModelWrapper):
    """In this example, a vehicle with 4 states and 2 actions, parks at (1, -1) heading to the top.
    We hope that the vechile finally parks at (0, 0) and heads to the right.
    The states and actions are listed as follows:

    - xu0(state): X position, limited to [-1, inf]
    - xu1(state): Y position
    - xu2(state): Heading angle
    - xu3(state): Velocity
    - xu4(action): Steering angle, limited to [-0.6, 0.6]
    - xu5(action): Acceleration, limited to [-3, 3]

    The dynamic model used in the example is introduced as follows.
    Given the time step :math:`h`, the rolling distance of the front wheels and the back wheels are given by

    .. math::
        \\begin{array}{rCl}
            f(v)&=&hv\\\\
            b(v,w)&=&d+f(v) \\cos(w)-\sqrt{d^2-f(v)^2 \\sin^2(w)},
        \\end{array}

    respectively. 

    Then, denote the the distance between the front axle and the back axle of the vehicle 
    by :math:`d`, the :\math:`h`-step dynamics of the vehicle are expressed as
    
    .. math::
        \\begin{array}{rCl}
            p_{x}(\\tau+1)&=&p_{x}(\\tau)+b\\big(v(\\tau), w(\\tau)\\big) \\cos \\big(\\theta(\\tau)\\big) \\nonumber\\\\
            p_{y}(\\tau+1)&=&p_{y}(\\tau)+b\\big(v(\\tau), w(\\tau)\\big) \\sin \\big(\\theta(\\tau)\\big) \\nonumber\\\\
            \\theta(\\tau+1)&=&\\theta(\\tau)+\\sin ^{-1}\\Big(f(v)\\sin \\big(w(\\tau)\\big)/d\\Big) \\nonumber\\\\
            v(\\tau+1)&=&v(\\tau)+h a(\\tau).
        \\end{array}

    Thus, the vehicle dynamic equation is given by

    .. math::
        \\begin{bmatrix}
            p_{x}(\\tau+1)\\\\
            p_{y}(\\tau+1)\\\\
            \\theta(\\tau+1)\\\\
            v(\\tau+1)
        \end{bmatrix} = 
        f\\left(
            \\begin{bmatrix}
                p_{x}(\\tau)\\\\
                p_{y}(\\tau)\\\\
                \\theta(\\tau)\\\\
                v(\\tau)
            \\end{bmatrix},
        \\begin{bmatrix}w(\\tau)\\\\a(\\tau)\\end{bmatrix} 
        \\right) 
        \\triangleq f\\big(x(\\tau),u(\\tau)\\big).

    :param is_with_constraints: Whether the box constraints of
        state and action variables are considered, defaults to True
    :type is_with_constraints: bool, optional
    :param T: Time horizon, defaults to 200
    :type T: int, optional
    """

    def __init__(self, is_with_constraints=True, T=200):
        ##### Dynamic Function ########
        h_constant = 0.1  # sampling time
        xu_var = sp.symbols('x_u:6')
        d_constant = 3
        h_d_constanT = h_constant/d_constant
        b_function = d_constant \
            + h_constant*xu_var[3]*sp.cos(xu_var[4]) \
            - sp.sqrt(d_constant**2 - (h_constant**2) *
                      (xu_var[3]**2)*(sp.sin(xu_var[4])**2))
        dynamic_function = sp.Array([
            xu_var[0] + b_function*sp.cos(xu_var[2]),
            xu_var[1] + b_function*sp.sin(xu_var[2]),
            xu_var[2] + sp.asin(h_d_constanT*xu_var[3]*sp.sin(xu_var[4])),
            xu_var[3]+h_constant*xu_var[5]])
        init_state = np.asarray([1, -1, np.pi/2, 0],
                                dtype=np.float64).reshape(-1, 1)
        if is_with_constraints:
            constr = np.asarray([[-1, np.inf], [-np.inf, np.inf],
                                 [-np.inf, np.inf], [-np.inf, np.inf], [-0.6, 0.6], [-3, 3]])
        else:
            constr = np.asarray([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf],
                                 [-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        ##### Objective Function ########
        switch_var = sp.symbols("s:2")
        add_param_obj = np.zeros((T, 2), dtype=np.float64)
        for tau in range(T):
            if tau < T-1:
                add_param_obj[tau] = np.asarray((1., 0.))
            else:
                add_param_obj[tau] = np.asarray((0., 1.))

        def Huber_fun(x, p):
            return sp.sqrt((x**2)+(p**2)) - p
        runing_obj = Huber_fun(
            xu_var[0], 0.01) + Huber_fun(xu_var[1], 0.01) + Huber_fun(xu_var[2], 0.01)
        terminal_obj = 1000*Huber_fun(xu_var[0], 0.01) + 1000*Huber_fun(
            xu_var[1], 0.01) + 1000*Huber_fun(xu_var[2], 0.01) + 100*Huber_fun(xu_var[3], 0.01)
        action_obj = xu_var[4]**2 + xu_var[5]**2
        obj_fun = switch_var[0] * runing_obj + \
            switch_var[1]*terminal_obj + action_obj
        super().__init__(dynamic_function=dynamic_function,
                         xu_var=xu_var,
                         constr=constr,
                         init_state=init_state,
                         T=T,
                         obj_fun=obj_fun,
                         add_param_var=switch_var,
                         add_param=add_param_obj)

    def render(self):
        """ This method will render an image for the current state."""
        if self._fig is None:
            super().create_plot(figsize=(5, 5), xlim=(-5, 10), ylim=(-7.5, 7.5))
            self._render_car = patches.FancyBboxPatch((0, 0), 3, 2, "round,pad=0.2")
            self._render_car.set_color('C0')
            self._ax.add_patch(self._render_car)
            plt.plot([-1, -1], [10, -10], 'C2')
            plt.plot([-1, 5], [2, 2], 'C2')
            plt.plot([-1, 5], [-2, -2], 'C2')
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
        """ The ``ActionSpace`` of the scenario."""
        return ActionSpace(action_range=[[-0.6, 0.6], [-3, 3]], action_type=["Continuous","Continuous"], action_info=["Steering Angle","Acceleration"])