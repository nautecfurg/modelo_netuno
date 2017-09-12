import tensorflow as tf
import architecture

class TransmissionNet(architecture.Architecture):
    def __init__(self):
        parameters_list = ['normalizer_parameters']

        self.config_dict = self.open_config(parameters_list)

    def prediction(self, sample, training=False):
        n_params = self.config_dict['normalizer_parameters']
        n_params.update({'is_training': training})
        #first convolution
        #INPUT: 16x16x3
        #KERNEL: 3x1
        #PADDING: 0
        #TIMES APPLIED: 16
        #OUTPUT: 14x16x12

        conv1 = tf.contrib.layers.conv2d(sample, 16, [3, 1], stride=(1, 1), padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=n_params, trainable=training)

        #second convolution
        #INPUT: 14x16x12
        #KERNEL: 1x3
        #PADDING: 0
        #TIMES APPLIED: 16
        #OUTPUT: 14x14x16

        conv2 = tf.contrib.layers.conv2d(conv1, 16, [1, 3], stride=(1, 1), padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=n_params, trainable=training)

        #third convolution
        #INPUT: 14x14x16
        #KERNEL: 5x1
        #PADDING: 0
        #TIMES APPLIED: 32
        #OUTPUT: 10x14x32

        conv3 = tf.contrib.layers.conv2d(conv2, 32, [5, 1], stride=(1, 1), padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=n_params, trainable=training)

        #fourth convolution
        #INPUT: 10x14x32
        #KERNEL: 1x5
        #PADDING: 0
        #TIMES APPLIED: 32
        #OUTPUT: 10x10x32

        conv4 = tf.contrib.layers.conv2d(conv3, 32, [1, 5], stride=(1, 1), padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=n_params, trainable=training)

        #first maxpool
        #INPUT: 10x10x32
        #KERNEL: 5x5
        #PADDING: 0
        #OUTPUT: 2x2x32

        pool1 = tf.contrib.layers.max_pool2d(conv4, [5, 5], stride=5, padding='VALID')
        print(pool1)

        #fifth convolution
        #INPUT: 2x2x32
        #KERNEL: 2x2
        #PADDING: 0
        #TIMES APPLIED: 1
        #OUTPUT: 1x1x1

        conv5 = tf.contrib.layers.conv2d(pool1, 1, [2, 2], stride=(1, 1), padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=n_params, trainable=training)

        one_constant = tf.constant(1)
        brelu = tf.minimum(tf.to_float(one_constant), tf.nn.relu(conv5, name="relu"), name="brelu")
        print(brelu)

        return brelu


    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
