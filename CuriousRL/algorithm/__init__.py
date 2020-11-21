import abc

class AlgoWrapper(object):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def init(self, scenario):
        pass

    @abc.abstractmethod
    def solve(self):
        pass

def algorithm(name: str, **kwargs) -> AlgoWrapper:
    if name == "BasiciLQR":
        from .algorithm.ilqr_solver.basic_ilqr import iLQRWrapper
        return iLQRWrapper(**kwargs)
    elif name == "LogBarrieriLQR":
        from .algorithm.ilqr_solver.advanced_ilqr import LogBarrieriLQR
        return LogBarrieriLQR(**kwargs)
    elif name == "NNiLQR":
        from .algorithm.ilqr_solver.advanced_ilqr import NNiLQR
        return NNiLQR(**kwargs)
    else:
        raise Exception("No algorithm \""+ name + "\"!")