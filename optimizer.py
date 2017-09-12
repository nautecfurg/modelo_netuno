#import numpy as np
#import tensorflow as tf
import json
import sys
import abc
import base

class Optimizer(base.Base):
    @abc.abstractmethod
    def minimize(self):
        pass

    # def verify_config(self, parameters_list):
    #     for parameter in parameters_list:
    #         if parameter not in self.config_dict:
    #             raise Exception('Config: ' + parameter + ' is necessary for ' +
    #                             self.__class__.__name__ + ' execution.')

    # def open_config(self, parameters_list):
    #     config_filename = sys.modules[self.__module__].__file__[:-3]+'.json'
    #     with open(config_filename) as config_file:
    #         self.config_dict = json.load(config_file)
    #     self.verify_config(parameters_list)
