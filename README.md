# CuriousRL

[TOC]

## scenario

The problem you want to use RL to solve. It can be a robot, a game, a dynamic model or anything.

class ScenarioWrapper

learn( )

play()



def Scenario(name, algo):

​	return ScenarioWrapper





```python
#只保存一些参数 网络啥的
policy1 = Policy() 
value1 = Value()
algo1 = algorithm("AC", policy = policy1, value = value1, ....)
#真正初始化所有东西
example1 = Scenario("package_name::envs_name", algo = algo1)
example1.learn()
```

```
#只保存一些参数 网络啥的
algo1 = algorithm("BasiciLQR",obj = obj, .....)
algo1 = iLQRWrapper()
#真正初始化所有东西
example1 = Scenario("dynamic::envs_name", algo = algo1)
example1.learn()
```





### OpenaiGym.py

class OpenaiGym(ScenarioWrapper)

### DynamicModel.py

def return_dynamic_model(“model_name”):

​	

class DynamicModel(ScenarioWrapper)

class vehicle(DynamicModel)

class cartpole(DynamicModel)

## data



## algorithm

class AlgoWrapper()

_init(self)



### iLQR

ClassicaliLQR(AlgoWrapper)

AdvancediLQR(AlgoWrapper)

### databased

## policy



## value



## utils



### Logger.py



### Network.py







### two_link_planar_manipulator

$$
x=[\theta_1\quad \dot \theta_1\quad \theta_2\quad \dot\theta_2\quad p_x\quad p_y]\\
u=[\ddot \theta_1\quad \ddot \theta_2]
$$

Then it follows that
$$
x_0(k+1)= x_0(k)+h x_1(k)\\
x_1(k+1)= x_1(k)+h u_1(k)\\
x_2(k+1)= x_2(k)+h x_3(k)\\
x_3(k+1)= x_3(k)+h u_2(k)\\
x_4(k+1)=l_1\cos(x_0(k))+l_2\cos(x_0(k)+x_2(k))\\
x_5(k+1)=l_1\sin(x_0(k))+l_2\sin(x_0(k)+x_2(k))
$$




