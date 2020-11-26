from abc import ABC, abstractmethod
class ScenarioWrapper(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def with_model(self):
        pass

    @abstractmethod
    def is_action_discrete(self):
        pass

    @abstractmethod
    def is_output_image(self):
        pass

    @abstractmethod
    def play(self):
        pass
