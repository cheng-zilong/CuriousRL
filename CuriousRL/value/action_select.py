from __future__ import annotations
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
import numpy as np
from typing import List

from CuriousRL.scenario import Scenario, ScenaroAsync
from CuriousRL.data import Data

class DiscreteActionSelect(object):
    @staticmethod
    def eps_greedy(net: nn.Module, scenario:Scenario, eps:float) -> List:
        if np.random.random() > eps:
            with torch.no_grad():
                x = scenario.curr_state.to(device=next(net.parameters()).device)
                if isinstance(scenario, ScenaroAsync):
                    action = net.forward(x.float()).max(1)[1].unsqueeze(1).tolist()
                else:
                    action = [net.forward(x.float().unsqueeze(0)).max(1)[1].item()]
        else:
            if isinstance(scenario, ScenaroAsync):
                num = len(scenario.action_space)
                action = []
                for i in range(num):
                    action += [scenario.action_space[i].sample()]
            else:
                action = scenario.action_space.sample()
        return action

class BoxActionSelect(object):
    pass