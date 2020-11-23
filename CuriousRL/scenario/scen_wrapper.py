import abc
class ScenarioWrapper(object):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def with_model(self):
        pass

    @abc.abstractmethod
    def is_action_discrete(self):
        pass

    @abc.abstractmethod
    def is_output_image(self):
        pass

    @abc.abstractmethod
    def play(self):
        pass
