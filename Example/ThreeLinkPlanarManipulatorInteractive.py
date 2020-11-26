import matplotlib.patches as patches
from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import ThreeLinkPlanarManipulator
import numpy as np
import matplotlib.pyplot as plt

class ThreeLinkPlanarManipulatorDemo(object):
    def __init__(self):
        self.scenario = ThreeLinkPlanarManipulator()
        fig, self.ax, _ = self.scenario.create_plot(xlim=(-6,6), ylim=(-6,6))
        self.algo = BasiciLQR(max_line_search=10) # self.algo = LogBarrieriLQR(max_line_search = 10)
        self.algo.init(self.scenario, is_save_json=False, is_use_logger=False)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.algo.solve()
        self.scenario.play()
        plt.show()

    def onclick(self, event):  
        self.ax.patches = []
        circle = patches.Circle((event.xdata, event.ydata), 0.1, alpha = 0.5)
        circle.set_color('C4')
        self.ax.add_patch(circle)
        add_param = self.algo.get_obj_add_param()
        add_param[:,0] = event.xdata*np.ones((self.algo.T))
        add_param[:,1] = event.ydata*np.ones((self.algo.T))
        self.algo.set_obj_add_param(add_param)
        self.algo.set_obj_fun_value(np.inf)
        self.algo.set_init_state(self.scenario.play_trajectory_current[0:self.scenario.get_n()].reshape(-1,1))
        self.algo.solve()
        self.scenario.play()

if __name__ == "__main__":
    ThreeLinkPlanarManipulatorDemo()