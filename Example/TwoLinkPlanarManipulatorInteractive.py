import matplotlib.patches as patches
from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import TwoLinkPlanarManipulator
import numpy as np
import matplotlib.pyplot as plt

class TwoLinkPlanarManipulatorDemo(object):
    def __init__(self):
        self.scenario = TwoLinkPlanarManipulator()
        fig, self.ax = self.scenario.create_plot(xlim=(-4,4), ylim=(-4,4))
        self.algo = BasiciLQR(max_line_search = 10) # self.algo = BasiciLQR(max_line_search = 10)
        self.algo.init(self.scenario)
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
        add_param[:,0] = event.xdata
        add_param[:,1] = event.ydata
        self.algo.set_obj_add_param(add_param)
        self.algo.set_obj_fun_value(np.inf)
        self.algo.set_init_state(self.scenario.play_trajectory_current[0:self.scenario.n].reshape(-1,1))
        self.algo.solve()
        self.scenario.play()

if __name__ == "__main__":
    TwoLinkPlanarManipulatorDemo()