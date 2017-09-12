import abc
import json
import sys


class Base(metaclass=abc.ABCMeta):

    def verify_config(self, parameters_list, config_dict):
        for parameter in parameters_list:
            if parameter not in config_dict:
                raise Exception('Config: ' + parameter + ' is necessary for ' +
                                self.__class__.__name__ + ' execution.')

    def open_config(self, parameters_list=[], config_filename=None):
        if config_filename is None:
            config_filename = sys.modules[self.__module__].__file__[:-3]+'.json'
        with open(config_filename) as config_file:
            config_dict = json.load(config_file)
        self.verify_config(parameters_list, config_dict)
        return config_dict
