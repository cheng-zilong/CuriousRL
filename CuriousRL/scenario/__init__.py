import abc

class ScenarioWrapper(object):
    def __init__(self, algo):
        self.algo = algo
        self.algo.init(self)
    
    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def play(self):
        pass

def scenario(name: str, **kwargs) -> ScenarioWrapper:
    package_name, scenario_name = name.split("::")
    if package_name == "Dynamic":
        import DynamicModel
        if scenario_name == "Vehicle":
            return DynamicModel.Vehicle(**kwargs) 
        else:
            raise Exception("No scenario_name \""+ name + "\" in package \"" + name + "\"!")
    elif package_name == "OpenaiGym":
        import OpenaiGym
        return OpenaiGym.ScenarioGym(scenario_name, **kwargs)
    else:
        raise Exception("No package \""+ name + "\"!")