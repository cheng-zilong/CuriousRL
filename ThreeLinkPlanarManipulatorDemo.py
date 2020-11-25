import matplotlib.patches as patches
from CuriousRL.algorithm import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import ThreeLinkPlanarManipulator
from CuriousRL import ProblemBuilderClass
import numpy as np
import matplotlib.pyplot as plt

class ThreeLinkPlanarManipulatorDemo(object):
    def __init__(self):
        self.scenario = ThreeLinkPlanarManipulator()
        fig, self.ax, _ = self.scenario.create_plot(xlim=(-6,6), ylim=(-6,6))
        self.algo = BasiciLQR() # self.algo = LogBarrieriLQR()
        self.problem = ProblemBuilderClass(self.scenario , self.algo)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.problem.learn(is_save_json=False)
        self.scenario.play()
        plt.show()

    def onclick(self, event):  
        self.ax.patches = []
        circle = patches.Circle((event.xdata, event.ydata), 0.1, alpha = 0.5)
        circle.set_color('C4')
        self.ax.add_patch(circle)
        add_param = self.problem.algo.obj_fun.get_add_param()
        add_param[:,0] = event.xdata*np.ones((self.problem.algo.T))
        add_param[:,1] = event.ydata*np.ones((self.problem.algo.T))
        self.problem.algo.obj_fun.update_add_param(add_param)
        self.problem.algo.obj_fun_value_last = np.inf
        self.problem.algo.set_init_state(self.scenario.play_trajectory_current[0:self.scenario.get_n()].reshape(-1,1))
        self.problem.learn(is_save_json=False)
        self.scenario.play()

if __name__ == "__main__":
    ThreeLinkPlanarManipulatorDemo()