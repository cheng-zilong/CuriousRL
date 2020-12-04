import matplotlib.patches as patches
from CuriousRL.algorithm.ilqr_solver import BasiciLQR, LogBarrieriLQR
from CuriousRL.scenario.dynamic_model import RoboticArmTracking
import numpy as np
import matplotlib.pyplot as plt

class RoboticArmTrackingDemo(object):
    def __init__(self):
        self.scenario = RoboticArmTracking()
        self.algo = LogBarrieriLQR(max_line_search = 10) # self.algo = BasiciLQR(max_line_search = 10)
        self.algo.init(self.scenario)
        self.algo.solve()
        self.scenario.render()
        self.scenario._fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.circle = patches.Circle((0, 0), 0.1, alpha = 0.5)
        self.circle.set_color('C4')
        self.scenario._ax.add_patch(self.circle)
        self.scenario.play()
        plt.show()

    def onclick(self, event):  
        self.circle.center = event.xdata, event.ydata
        add_param = self.algo.get_obj_add_param()
        add_param[:,0] = event.xdata
        add_param[:,1] = event.ydata
        self.algo.set_obj_add_param(add_param)
        self.algo.set_obj_fun_value(np.inf)
        self.algo.set_init_state(self.scenario._current_state[0:self.scenario.n].reshape(-1,1))
        self.algo.solve()
        self.scenario.play()

if __name__ == "__main__":
    RoboticArmTrackingDemo()