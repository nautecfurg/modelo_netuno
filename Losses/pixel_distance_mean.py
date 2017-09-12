import tensorflow as tf

import loss


class PixelDistanceMean(loss.Loss):
    def __init__(self):
        parameters_list = []
        self.config_dict = self.open_config(parameters_list)

    def evaluate(self, architecture_output, target_output):
        return tf.reduce_mean(
            tf.sqrt(
                tf.reduce_sum(
                    tf.squared_difference(architecture_output, target_output),
                    reduction_indices=[3])),
            reduction_indices=[1, 2])
