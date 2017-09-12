#import numpy as np
#import tensorflow as tf
import abc
import base

class DatasetManager(base.Base):
    '''
    Abstract class designed for loading and storing
    objects from memory. 
    '''
    


    @abc.abstractmethod
    def convert_data(self):
        """Convert Data

        responsible for loading datasets and storing in the proper way.
        Usually it will be stored in tfrecords.

        Args:
            self: the instance

        """
        pass

    