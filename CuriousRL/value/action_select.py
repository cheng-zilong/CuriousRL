from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import List

from CuriousRL.scenario import Scenario

class DiscreteActionSelect(object):
    @staticmethod
    def eps_greedy(net: nn.Module, scenario:Scenario, eps:float) -> List:
        if np.random.random() > eps:
            with torch.no_grad():
                x = scenario.elem.next_state.to(device=next(net.parameters()).device)
                if scenario.mode == "single":
                    action = [net.forward(x.float().unsqueeze(0)).max(1)[1].item()]
                elif scenario.mode == "multiple":
                    action = net.forward(x.float()).max(1)[1].unsqueeze(1).tolist()
                else:
                    action = None
                    raise Exception("No \"%s\" mode"%(scenario.mode))
        else:
            if scenario.mode == "single":
                action = scenario.action_space.sample()
            elif scenario.mode == "multiple":
                num = len(scenario.action_space)
                action = []
                for i in range(num):
                    action += [scenario.action_space[i].sample()]
            else:
                action = None
                raise Exception("No \"%s\" mode"%(scenario.mode))
        return action

class BoxActionSelect(object):
    pass