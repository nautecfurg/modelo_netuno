import architecture
import tensorflow as tf
import numpy as np

class AffinityBranch(architecture.Architecture):
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                "validation_period", "model_saving_period", "radius"]
        self.config_dict = self.open_config(parameters_list)
        width = self.config_dict['input_size'][0]
        height = self.config_dict['input_size'][1]
        radius = self.config_dict['radius']
        mask = []
        for i in range(width):
            for j in range(height):
                y, x = np.ogrid[-i:width-i, -j:height-j]
                mask.append(x*x + y*y <= radius)
        mask = np.array(mask, dtype=np.int8)
        self.mask = tf.constant(mask)
    def prediction(self, sample, training=False, num_classes=2):
        input_size = self.config_dict["input_size"]
        #raw_F = np.abs(np.subtract.outer(a, a)) TODO: fazer isso no tf

        streched_samples = tf.reshape(sample, [
            input_size[0] * input_size[1] * input_size[2], :])
