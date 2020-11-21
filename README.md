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
algo1 = algorithm("iLQR",obj = obj, .....)
#真正初始化所有东西
example1 = Scenario("package_name::envs_name", algo = algo1)
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



