from abc import ABC, abstractmethod
class ScenarioWrapper(ABC):
    @abstractmethod
    def with_model(self):
        pass

    @abstractmethod
    def is_action_discrete(self):
        pass

    @abstractmethod
    def is_output_image(self):
        pass

    def play(self):
        raise NotImplementedError
