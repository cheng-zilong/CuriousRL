#%%
import numpy as np
import sympy as sp
from datetime import datetime
from CuriousRL.utils.Logger import logger
from .model_wrapper import DynamicModelWrapper
from .obj_fun import ObjectiveFunctionWrapper


class VehicleTracking(DynamicModelWrapper):
    """ In this example, the vehicle packing at 0, 0, heading to the postive direction of the y axis
        We hope the vechile can tracking the reference y=-10 with the velocity 8, and head to the right
    """
    def __init__(self, algo):
        self.T = 100
        self.n = 4
        self.m = 2
        h_constant = 0.1 # sampling time
        x_u = sp.symbols('x_u:6')
        d_constanT = 3
        h_d_constanT = h_constant/d_constanT
        b_function = d_constanT \
                    + h_constant*x_u[3]*sp.cos(x_u[4]) \
                    -sp.sqrt(d_constanT**2 
                        - (h_constant**2)*(x_u[3]**2)*(sp.sin(x_u[4])**2))
        system = sp.Array([  
                    x_u[0] + b_function*sp.cos(x_u[2]), 
                    x_u[1] + b_function*sp.sin(x_u[2]), 
                    x_u[2] + sp.asin(h_d_constanT*x_u[3]*sp.sin(x_u[4])), 
                    x_u[3]+h_constant*x_u[5]
                ])
        init_state = np.asarray([0,0,np.pi/2,0],dtype=np.float64).reshape(-1,1)
        init_input = np.zeros((self.T,self.m,1))
        # Objective function
        C_matrix = np.diag([0.,1.,1.,1.,10.,10.])
        r_vector = np.asarray([0.,-10.,0.,8.,0.,0.])
        obj_fun = ObjectiveFunctionWrapper((x_u - r_vector)@C_matrix@(x_u - r_vector), x_u)
        super().__init__(algo, system, x_u, init_state, init_input, obj_fun, "DynamicModel: VehicleTracking", None, None)

    def learn(self, file_name = None):
        
        self.algo.print_params()
        self.algo.solve()



def cart_pole(h_constant = 0.02):
    m_c = 1 # car mass
    m_p = 0.1 # pole mass
    l=0.5 # half pole length
    h = 0.02 # sampling time
    g = 9.8 # gravity
    # x0: theta 
    # x1: dot_theta 
    # x2: x 
    # x3: dot_x 
    # x4: F
    x_u = sp.symbols('x_u:5') 
    gamma = (x_u[4] + m_p*l*(x_u[1]**2)*sp.sin(x_u[0]))/(m_c+m_p)
    dotdot_theta = (g*sp.sin(x_u[0])-sp.cos(x_u[0])*gamma)/(l*((4/3)-((m_p*(sp.cos(x_u[0])**2))/(m_c+m_p))))
    dotdot_x = gamma - (m_p*l*dotdot_theta*sp.cos(x_u[0]))/(m_c+m_p)
    system = sp.Array([  
        x_u[0] + h*x_u[1],
        x_u[1] + h*dotdot_theta,
        x_u[2] + h*x_u[3],
        x_u[3] + h*dotdot_x
    ])
    return system, x_u, 4, 1

def cart_pole_advanced(h_constant = 0.02):
    m_c = 1 # car mass
    m_p = 0.1 # pole mass
    l=0.5 # half pole length
    h = 0.02 # sampling time
    g = 9.8 # gravity
    # x0: x 
    # x1: dot_x 
    # x2: sin theta
    # x3: cos theta
    # x4: dot_theta
    # x5: F
    x_u = sp.symbols('x_u:6') 
    theta_next = sp.atan2(x_u[2], x_u[3]) + h * x_u[4]
    gamma = (x_u[5] + m_p*l*(x_u[4]**2)*x_u[2])/(m_c+m_p)
    dotdot_theta = (g*x_u[2]-x_u[3]*gamma)/(l*((4/3)-((m_p*(x_u[3]**2))/(m_c+m_p))))
    dotdot_x = gamma - (m_p*l*dotdot_theta*x_u[3])/(m_c+m_p)
    system = sp.Array([  
        x_u[0] + h*x_u[1],
        x_u[1] + h*dotdot_x,
        sp.sin(theta_next),
        sp.cos(theta_next),
        x_u[4] + h*dotdot_theta
    ])
    return system, x_u, 5, 1