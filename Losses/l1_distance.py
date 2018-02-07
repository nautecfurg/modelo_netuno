import loss
import tensorflow as tf
class L1Distance(loss.Loss):
    def __init__(self):
        parameters_list = []
        self.config_dict = self.open_config(parameters_list)
    def evaluate(self, architecture_input, architecture_output, target_output):
        return tf.reduce_mean(tf.abs(architecture_output - target_output))
