import abc

class Evaluate(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def eval(self, opt_values, architecture_instance):
        """This is a abstract method for evaluate implementation.

        """
        pass
