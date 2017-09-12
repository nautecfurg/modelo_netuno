import abc
import sys
import json
import base

class Loss(base.Base):

    @abc.abstractmethod
    def evaluate(self, architecture_output, target_output):
        """This is a abstract method for defining loss functions.

        Each loss must implement this method. Depending on the
        architecture the output shape varies. Depending on
        the output shape a determined loss can or not be used.

        Args:
            architecture_output: architecture output tensor
            target_output: desired output must have the same
            shape as architecture_output
        Returns:
            loss output:

        """
        pass

    def train(self):
        pass

    def trainable(self):
        return False
    
    def verify_config(self, parameters_list):
        for parameter in parameters_list:
            if parameter not in self.config_dict:
                raise Exception('Config: ' + parameter + ' is necessary for ' +
                                self.__class__.__name__ + ' execution.')

    def open_config(self, parameters_list):
        if parameters_list: # if parameters list is empty does not open file
            config_filename = sys.modules[self.__module__].__file__[:-3]+'.json'
            with open(config_filename) as config_file:
                self.config_dict = json.load(config_file)
            self.verify_config(parameters_list)
