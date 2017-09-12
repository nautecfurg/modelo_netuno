import loss
import tensorflow as tf
class L2Distance(loss.Loss):
    def __init__(self):
        parameters_list = []
        self.config_dict = self.open_config(parameters_list)
    def evaluate(self, architecture_output, target_output):
        return tf.sqrt(tf.reduce_mean(tf.squared_difference
                                     (architecture_output, target_output)))
