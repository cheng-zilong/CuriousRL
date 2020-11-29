from __future__ import annotations
import torch
from torch import tensor, Tensor
from typing import Union

class Data(object):
    """This is the class for building the reinforcement learning data inlcuding observations, actions, rewards, and done flags.
    This is the basic data class for all the algorithms in the CuriousRL package. All data commnunications in the algorithms are based
    on this class. Each ``Data`` instance can contain one or several observation(s), action(s), reward(s), and done flag(s).

    There are **two** ways to initial a Data class. The first way initializes the ``Data`` class from a **Singleton Data**. 
    The Second way initializes the ``Data`` class from **normal data**. 
    
    In terms of the **singleton data**, the type of reward is float and the type of done_flag is bool. In this case,
    the observations and actions are given directly without the index of data. For example,
    if an observation is a grey image, the type of the observation is Tensor[512,512], the type of action is Tensor[5],
    the type of reward is float, and the type of done flag is bool.  
    
    In terms of the **normal data**, the types of reward and done_flag are both Tensor with the index dimension as the first dimension. 
    In this case, the observations, actions, rewards, done_flags are all given with the first dimension as the index of the data. For example,
    if the observation is a grey image, then the type of observation is Tensor[1,512,512], the type of action is Tensor[1,5],
    the type of reward is Tensor[1,1], and the type of done_flag is Tensor[1,1]. In this case, one ``Data`` instance can contain many pieces of data. 
    For example, if the observations are 10 grey images, then the type of observations is Tensor[10,512,512], the type of actions is Tensor[10,5],
    the type of rewards is Tensor[10,1], and the type of done_flag is Tensor[10,1].

    :param obs: Observations
    :type obs: Tensor
    :param action: Actions
    :type action: Tensor
    :param reward: Rewards
    :type reward: Union[Tensor, float]
    :param done_flag: The flag deciding whether one episode is done.
    :type done_flag: Union[Tensor, bool]
    """
    def __init__(self, obs:Tensor, action:Tensor, reward:Union[Tensor, float], done_flag:Union[Tensor, bool]):
        # data type checking
        if (not isinstance(reward, Tensor)) and (not isinstance(reward, float)):
            raise Exception("Reward must be in the type of Tensor or float!")
        if (not isinstance(done_flag, Tensor)) and (not isinstance(done_flag, bool)):
            raise Exception("Done flag must be in the type of Tensor or bool!")
        if isinstance(reward, float) and isinstance(done_flag, bool):
            self._obs = torch.unsqueeze(obs, 0)
            self._action = torch.unsqueeze(action, 0)
            if all((obs.is_cuda, action.is_cuda)):
                self._reward = torch.Tensor([reward]).cuda()
                self._done_flag = torch.Tensor([done_flag]).cuda()
                self._is_cuda = True
            elif all((not obs.is_cuda, not action.is_cuda)):
                self._reward = torch.Tensor([reward])
                self._done_flag = torch.Tensor([done_flag])
                self._is_cuda = False
            else:
                raise Exception("The devices of reward and done flag do not fit!")
        elif isinstance(reward, Tensor) and isinstance(done_flag, Tensor):
            if all((obs.is_cuda, action.is_cuda, reward.is_cuda, done_flag.is_cuda)):
                self._is_cuda = True
            elif all((not obs.is_cuda, not action.is_cuda, not reward.is_cuda, not done_flag.is_cuda)):
                self._is_cuda = False
            else:
                raise Exception("The devices of observation, action, reward and done flag do not fit!")
            self._obs  = obs
            self._action = action
            self._reward = reward
            self._done_flag = done_flag
        else:
            raise Exception("The type of reward and the type of done flag do not fit!")
    
    def __len__(self):
        return self._obs.shape[0]

    def __str__(self):
        return "Observation:\n" + str(self._obs) + "\n"\
            +  "Action:\n" + str(self._action) + "\n"\
            +  "Reward:\n" + str(self._reward) + "\n"\
            +  "Done Flag:\n" + str(self._done_flag) 

    @property
    def is_cuda(self):
        return self._is_cuda

    def get_obs(self) -> Tensor:
        """Get observation

        :return: observation
        :rtype: Tensor[data_size, *obs_dim]
        """
        return self._obs

    def get_action(self)-> Tensor:
        """Get action

        :return: action
        :rtype: Tensor[data_size, action_dim]
        """
        return self._action

    def get_reward(self) -> Tensor:
        """Get reward

        :return: reward
        :rtype: Tensor[data_size]
        """
        return self._reward

    def get_done_flag(self) -> Tensor:
        """Get done flag

        :return: Done flag 
        :rtype: Tensor[data_size]
        """
        return self._done_flag

    def clone(self) -> Data:
        new_data = Data(self._obs.clone(), self._action.clone(), self._reward.clone(), self._done_flag.clone())
        return new_data

    def cat(self, datas:Tuple[Data, ...]) -> Data:
        if isinstance(datas, tuple):
            if all((self.is_cuda, *[data.is_cuda for data in datas])) or (not any((self.is_cuda, *[data.is_cuda for data in datas]))):
                return _cat((self, *datas))
            else:
                raise Exception("The devices of the Data instances are not consistent!")
        else:
            if self.is_cuda == datas.is_cuda:
                return _cat((self, datas))
            else:
                raise Exception("The devices of the Data instances are not consistent!")

def _cat(datas:Tuple[Data, ...]) -> Data:
    new_data = datas[0].clone()
    for data in datas[1:]:
        new_data._obs = torch.torch.cat([new_data._obs, data._obs], dim=0)
        new_data._action = torch.torch.cat([new_data._action, data._action], dim=0)
        new_data._reward = torch.torch.cat([new_data._reward, data._reward], dim=0)
        new_data._done_flag = torch.torch.cat([new_data._done_flag, data._done_flag], dim=0)
    return new_data
    
