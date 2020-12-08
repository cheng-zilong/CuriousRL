from __future__ import annotations
import numpy as np
import sympy as sp
from numba import njit
from CuriousRL.scenario.scenario_wrapper import Scenario
from CuriousRL.utils.Logger import logger
from CuriousRL.utils.config import global_config
import matplotlib.pyplot as plt
from CuriousRL.data import Data
from typing import TYPE_CHECKING, List
import torch
from torch import Tensor, tensor
import sys

class DynamicModel(Scenario):
    """This is a class for creating dynamic models from the state transfer function 
    given in the form :math:`x(k+1)=f\\big (x(k),u(k)\\big)`, where :math:`x` is the state, and :math:`u`
    is the action. In each dynamic model scenario, an objective function is required to be given in the form
    :math:`J=\\sum_{\\tau=0}^{T} J_\\tau`. 

    The additional variable are used to design the time variant objective functions. For example, for the objective function 
    with the terminal penality, the weighing parameters are changed in the last time stamp. To be more specific, if a linear
    quadratic objective function is considered, :math:`J=(x-r)^TQ(x-r)`, then we can define :math:`Q` as time variant. In the 
    ``cart_pole_swingup2`` example, the following codes are used. 

    ::

        C_matrix_diag = sp.symbols("c:6")
        r_vector = np.asarray([0, 0, 0, 1, 0, 0])
        add_param_obj = np.zeros((T, 6), dtype = np.float64)
        for tau in range(T):
            if tau < T-1:
                add_param_obj[tau] = np.asarray((0.1, 0.1, 1, 1, 0.1, 1))
            else: 
                add_param_obj[tau] = np.asarray((0.1, 0.1, 10000, 10000, 1000, 0))
        obj_fun = (xu_var- r_vector)@np.diag(np.asarray(C_matrix_diag))@(xu_var- r_vector)

    Also, in the tracking problem with a time variant reference, the additional parameter also can be used. In the example of 
    ``two_link_planar_manipulator``, the reference is different in each iLQR optimization, therefore we use the additional parameters. 

    ::

        position_var = sp.symbols("p:2") # x and y
        C_matrix = np.diag([0.,      10.,     0.,        10.,          10000,                             10000,                 1,           1])
        r_vector = np.asarray([0.,       0.,     0.,         0.,          position_var[0],            position_var[1],              0.,          0.])
        obj_fun = (xu_var - r_vector)@C_matrix@(xu_var - r_vector) 

    If the additional parameters are not used, leave them to be None. 

    :param dynamic_function: State transfer function :math:`f(x(k),u(k))`
    :type dynamic_function: symbolic arrawy in sympy
    :param xu_var: State and action variables
    :type xu_var: Tuple[sympy.symbol, ...]
    :param constr: Constraint of each state and action variable
    :type constr: numpy.array
    :param init_state: Initial state of the dynamic system
    :type init_state: numpy.array
    :param T: Time horizon
    :type T: int
    :param obj_fun: Objective function :math:`J_\\tau`
    :type obj_fun: symbolic arrawy in sympy
    :param add_param_var: Additional parameter variable in the objective function
    :type add_param_var: Tuple[sympy.symbol, ...], optional
    :param add_param: Additional parameter with the first index as time stamp, defaults to None
    :type add_param: numpy.array, optional
    """
    def __init__(self, 
                dynamic_function: sp.ImmutableDenseNDimArray, 
                xu_var: Tuple[sp.Array, ...], 
                constr,
                init_state, 
                T,
                obj_fun,
                add_param_var = None,
                add_param = None):
        self._n = int(init_state.shape[0])
        self._m = int(len(xu_var) - self._n)
        self._T = T
        if add_param_var is None:
            add_param_var = sp.symbols("no_use")
            add_param = np.zeros((self._T, 1))
        self._dynamic_function = dynamic_function
        self._dynamic_function_lamdify = njit(sp.lambdify([xu_var], dynamic_function, "math"))
        self._obj_fun_lamdify = njit(sp.lambdify([xu_var, add_param_var], obj_fun, "math"))
        self._xu_var = xu_var
        self._init_state = init_state
        self._constr = constr
        self._obj_fun = obj_fun
        self._add_param_var = add_param_var
        self._add_param = add_param
        self._current_state = init_state[:,0]
        self._tau = 0
        self._fig = None
        self._ax = None
        super().__init__(n = self._n,
                        m = self._m,
                        T = self._T,
                        dynamic_function = dynamic_function,
                        xu_var = xu_var,
                        constr = constr,
                        init_state = init_state,
                        obj_fun = obj_fun,
                        add_param_var = add_param_var,
                        add_param = add_param)

    def _create_plot(self, figsize =(5, 5), xlim = (-6,6), ylim = (-6,6)):
        """Create a plot for annimation.

        :param figsize: Annimation figure size, defaults to (5, 5)
        :type figsize: Tuple[float, float], optional
        :param xlim: Limits to x-axis, defaults to (-6,6)
        :type xlim: Tuple[float, float], optional
        :param ylim: Limits to y-axis, defaults to (-6,6)
        :type ylim: Tuple[float, float], optional
        """
        if self._fig == None:
            logger.info("[+] Annimation figure is created!")
            self._fig = plt.figure(figsize = figsize)
            self._ax = self._fig.add_subplot(111) 
            self._ax.axis('equal')
            self._ax.set_xlim(*xlim)
            self._ax.set_ylim(*ylim)
            self._ax.grid(True)
            self._fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_close(self, event):
        self._fig = None
        self._ax = None
        logger.info("[+] Annimation figure is closed!")
        sys.exit()

    @property
    def dynamic_function(self):
        """State transfer function in the type of sympy symbolic array."""
        return self._dynamic_function

    @property
    def obj_fun(self):
        """Objective function in the type of sympy symbolic array."""
        return self._obj_fun

    @property
    def xu_var(self):
        """Stata and action variables in the type of sympy symbols."""
        return self._xu_var
        
    @property
    def init_state(self):
        """Initial state of the dynamic system in the type of numpy array."""
        return self._init_state

    @property
    def n(self):
        """Number of state variables."""
        return self._n

    @property
    def m(self):
        """Number of action variables."""
        return self._m

    @property
    def T(self):
        """Time horizon."""
        return self._T

    @property
    def add_param_var(self):
        """Additional parameter variables."""
        return self._add_param_var

    @property
    def add_param(self):
        """Additional parameters."""
        return self._add_param 
    
    @property
    def constr(self):
        """Constraints of state and action variables."""
        return self._constr

    def reset(self) -> Tensor:
        """Reset the current state to the initial state."""
        self._tau = 0
        self._current_state = self._init_state[:,0]
        return self.state

    def step(self, action: List) -> Data:
        """Evaulate the next state given an action. Return state, action, next_state, reward, done_flag in a ``Data`` instance."""
        self._tau += 1
        last_state = self.state 
        self._current_state = self._dynamic_function_lamdify(np.concatenate([self._current_state, action]))
        for i, c in enumerate(self._constr[:self._n]):
            self._current_state[i] = min(max(c[0], self._current_state[i]), c[1]) 
        self._reward = -self._obj_fun_lamdify(np.concatenate([self._current_state, action]), self._add_param[self._tau-1])
        if self._tau == self._T:
            done_flag = True
            action = tensor(action, dtype=torch.float).flatten()
            data = Data(state=last_state,
                action=action,
                next_state=self.state,
                reward=self.reward,
                done_flag=done_flag)
            self.reset()
        else:
            done_flag = False
            action = tensor(action, dtype=torch.float).flatten()
            data = Data(state=last_state,
                action=action,
                next_state=self.state,
                reward=self.reward,
                done_flag=done_flag)
        return data

    def play(self, logger_folder=None, no_iter=-1):
        """ This method will play an animation for a whole episode.
        If ``logger_folder`` exists and a json file inside the folder containing results is saved, 
        then the specific iteration can be chosen for the animation.
        If ``logger_folder`` is set to be none, 
        then the trajectroy in the last iteration will be played.

        :param logger_folder: Name of the logger folder where the json file is saved, defaults to None
        :type logger_folder: str, optional
        :param no_iter: Number of iteration for the animation. 
            If it is set as -1, then the trajectroy in the last iteration will be played. defaults to -1.
        :type no_iter: int, optional
        """
        trajectory = np.asarray(logger.read_from_json(
            logger_folder, no_iter)["trajectory"])
        self._is_interrupted = False
        for i in range(self.T):
            self._current_state = trajectory[i,:,0]
            if self._fig is not None:
                self._fig.canvas.set_window_title("Time:" + str(i))
            self.render()
            if self._is_interrupted:
                return
        self._is_interrupted = True

    @property
    def state(self) -> Tensor:
        if global_config.is_cuda:
            return tensor(self._current_state, dtype=torch.float).flatten().cuda()
        else:
            return tensor(self._current_state, dtype=torch.float).flatten()

    @property
    def reward(self) -> float:
        return self._reward