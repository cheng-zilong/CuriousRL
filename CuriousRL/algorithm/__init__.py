import abc

class AlgoWrapper(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    @abc.abstractmethod
    def init(self, scenario):
        pass

    @abc.abstractmethod
    def solve(self):
        pass

def algo(name: str, **kwargs) -> AlgoWrapper:
    if name == "BasiciLQR":
        from .ilqr_solver.basic_ilqr import BasiciLQR
        return iLQRWrapper(**kwargs)
    elif name == "LogBarrieriLQR":
        from .ilqr_solver.advanced_ilqr import LogBarrieriLQR
        return LogBarrieriLQR(**kwargs)
    elif name == "NNiLQR":
        from .ilqr_solver.advanced_ilqr import NNiLQR
        return NNiLQR(**kwargs)
    else:
        raise Exception("No algorithm \""+ name + "\"!")