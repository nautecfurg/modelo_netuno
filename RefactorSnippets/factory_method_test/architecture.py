import abc

class Architecture(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def prediction(self):
        pass
