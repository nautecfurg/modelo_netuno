#import numpy as np
#import tensorflow as tf
import json
import sys
import abc
import base

class Architecture(base.Base):
    @abc.abstractmethod
    def prediction(self, sample, training=False):
        """This is a abstract method for architectures prediction.

        Each architecture must implement this method. Depending on
        each diferent implementation the output shape varies. So
        the loss must be chosen acoording with the achitecture
        implementation.
        In a similar way the architecture implementation depends on
        the dataset shape.

        Args:
            sample: networks input tensor
            training: boolean value indication if this prediction is
            being used on training or not

        Returns:
            achitecture output: networks output tensor

        """
        pass

    @abc.abstractmethod
    def get_validation_period(self):
        pass

    @abc.abstractmethod
    def get_model_saving_period(self):
        pass

    @abc.abstractmethod
    def get_summary_writing_period(self):
        pass

    def get_layer(self, layer_name):
        """This method returns a reference to a layer in the architecture.
        It must be overridden if the user wishes to visualize the hidden
        layers of the network, but doesn't need to be implemented otherwise.

        Args:
            layer_name: The name of the desired layer


        Returns:
            layer: a reference to the layer's tensor

        """
        layer=None
        return layer

    # def verify_config(self, parameters_list, config_dict):
    #     for parameter in parameters_list:
    #         if parameter not in config_dict:
    #             raise Exception('Config: ' + parameter + ' is necessary for ' +
    #                             self.__class__.__name__ + ' execution.')

    # def open_config(self, parameters_list=[], config_filename=None):
    #     if config_filename is None:
    #         config_filename = sys.modules[self.__module__].__file__[:-3]+'.json'
    #     with open(config_filename) as config_file:
    #         config_dict = json.load(config_file)
    #     self.verify_config(parameters_list, config_dict)
    #     return config_dict
