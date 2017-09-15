import architecture
import tensorflow as tf
import Architectures.Layers.guidedfilter_color as gc

class MscnnGuidedColor(architecture.Architecture):
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False):
        " Coarse-scale Network"
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        conv1 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=5, kernel_size=[11, 11],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1, kernel_size=[2, 2], stride=2,
                                             padding='VALID')    #pooling

        upsamp1 = tf.image.resize_nearest_neighbor(pool1, self.input_size)    # upsampling


        conv2 = tf.contrib.layers.conv2d(inputs=upsamp1, num_outputs=5, kernel_size=[9, 9],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool2 = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=[2, 2], stride=2,
                                             padding='VALID') #pooling

        upsamp2 = tf.image.resize_nearest_neighbor(pool2, self.input_size)    # upsampling

        conv3 = tf.contrib.layers.conv2d(inputs=upsamp2, num_outputs=10, kernel_size=[7, 7],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool3 = tf.contrib.layers.max_pool2d(inputs=conv3, kernel_size=[2, 2], stride=2,
                                             padding='VALID') #pooling

        upsamp3 = tf.image.resize_nearest_neighbor(pool3, self.input_size)    # upsampling

        linear_combination = tf.contrib.layers.conv2d(inputs=upsamp3, num_outputs=1,
                                                      kernel_size=[1, 1],
                                                      stride=[1, 1], padding='SAME',
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params=normalizer_params,
                                                      activation_fn=tf.nn.sigmoid)

        """Fine-scale Network"""

        conv4 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=4, kernel_size=[7, 7],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool4 = tf.contrib.layers.max_pool2d(inputs=conv4, kernel_size=[2, 2], stride=2,
                                             padding='VALID') #pooling e upsampling

        upsamp4 = tf.image.resize_nearest_neighbor(pool4, self.input_size)    # upsampling

        concatenation = tf.concat([upsamp4, linear_combination], 3)

        conv5 = tf.contrib.layers.conv2d(inputs=concatenation, num_outputs=5, kernel_size=[5, 5],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool5 = tf.contrib.layers.max_pool2d(inputs=conv5, kernel_size=[2, 2], stride=2,
                                             padding='VALID') #pooling e upsampling

        upsamp5 = tf.image.resize_nearest_neighbor(pool5, self.input_size)    # upsampling

        conv6 = tf.contrib.layers.conv2d(inputs=upsamp5, num_outputs=10, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool6 = tf.contrib.layers.max_pool2d(inputs=conv6, kernel_size=[2, 2], stride=2,
                                             padding='VALID')  # pooling e upsampling

        upsamp6 = tf.image.resize_nearest_neighbor(pool6, self.input_size)    # upsampling

        linear_combination2 = tf.contrib.layers.conv2d(inputs=upsamp6, num_outputs=1,
                                                       kernel_size=[1, 1], stride=[1, 1],
                                                       padding='SAME',
                                                       normalizer_fn=tf.contrib.layers.batch_norm,
                                                       normalizer_params=normalizer_params,
                                                       activation_fn=tf.nn.sigmoid)

        guided_trans = gc.guidedfilter_color(sample, linear_combination2, r=20, eps=10**-3)
        tf.summary.image("architecture_output", guided_trans)
        return guided_trans



    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
