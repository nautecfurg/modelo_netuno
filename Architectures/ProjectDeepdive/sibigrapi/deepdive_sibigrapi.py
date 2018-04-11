import architecture
import tensorflow as tf
import Architectures.Layers.inception_resnet_a as ira
import Architectures.Layers.inception_resnet_b as irb
import Architectures.Layers.inception_resnet_c as irc

class DeepdiveSibigrapi(architecture.Architecture):
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False):
        " Coarse-scale Network"
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        conv1 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=16, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
        
        module_a = ira.inception_resnet_a(conv1, normalizer_params)
        module_b = irb.inception_resnet_b(module_a, normalizer_params)
        module_c = irc.inception_resnet_c(module_b, normalizer_params)

        conv2 = tf.contrib.layers.conv2d(inputs=module_c, num_outputs=3, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        const_1 = tf.constant(1, dtype=tf.float32)
        brelu = tf.minimum(const_1, tf.nn.relu(conv2))
        tf.summary.image("architecture_output", brelu)
        return brelu



    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
