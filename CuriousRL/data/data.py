from __future__ import annotations
import torch
from torch import tensor

class Data(object):
    def __init__(self, obs:tensor, action:tensor, reward:tensor, done_flag:tensor):
        self._obs  = obs
        self._action = action
        self._reward = reward
        self._done_flag = done_flag
    
    def __len__(self):
        return self._obs.shape[0]

    def __str__(self):
        return "Observation:\n" + str(self._obs) + "\n"\
            +  "Action:\n" + str(self._action) + "\n"\
            +  "Reward:\n" + str(self._reward) + "\n"\
            +  "Done Flag:\n" + str(self._done_flag) 

    def get_obs(self) -> tensor:
        """Get observation

        :return: observation
        :rtype: tensor[data_size, *obs_dim]
        """
        return self._obs

    def get_action(self)-> tensor:
        """Get action

        :return: action
        :rtype: tensor[data_size, action_dim]
        """
        return self._action

    def get_reward(self) -> tensor:
        """Get reward

        :return: reward
        :rtype: tensor[data_size]
        """
        return self._reward

    def get_done_flag(self) -> tensor:
        """Get done flag

        :return: Done flag 
        :rtype: tensor[data_size]
        """
        return self._done_flag

    def clone(self) -> Data:
        new_data = Data(self._obs.clone(), self._action.clone(), self._reward.clone(), self._done_flag.clone())
        return new_data

    def cat(self, datas:Tuple[Data, ...]) -> Data:
        return _cat((self, *datas))
        

def _cat(datas:Tuple[Data, ...]) -> Data:
    new_data = datas[0].clone()
    for data in datas[1:]:
        new_data._obs = torch.torch.cat([new_data._obs, data._obs], dim=0)
        new_data._action = torch.torch.cat([new_data._action, data._action], dim=0)
        new_data._reward = torch.torch.cat([new_data._reward, data._reward], dim=0)
        new_data._done_flag = torch.torch.cat([new_data._done_flag, data._done_flag], dim=0)
    return new_data
    
