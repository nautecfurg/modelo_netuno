import architecture
import tensorflow as tf

class ArchitectureDepthTest(architecture.Architecture):
    def __init__(self):
        parameters_list = ['example', 'input_size', 'output_size',
                           'depth_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)

    def prediction(self, sample, training=False):
        " Coarse-scale Network"
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        conv1 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=5, kernel_size=[11, 11],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1, kernel_size=[2, 2], stride=1,
                                             padding='SAME') #pooling e upsampling


        conv2 = tf.contrib.layers.conv2d(inputs=pool1, num_outputs=5, kernel_size=[9, 9],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool2 = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=[2, 2], stride=1,
                                             padding='SAME') #pooling e upsampling

        conv3 = tf.contrib.layers.conv2d(inputs=pool2, num_outputs=10, kernel_size=[7, 7],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool3 = tf.contrib.layers.max_pool2d(inputs=conv3, kernel_size=[2, 2], stride=1,
                                             padding='SAME') #pooling e upsampling

        linear_combination = tf.contrib.layers.conv2d(inputs=pool2, num_outputs=3,
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

        pool4 = tf.contrib.layers.max_pool2d(inputs=conv4, kernel_size=[2, 2], stride=1,
                                             padding='SAME') #pooling e upsampling

        concatenation = tf.concat([pool4, linear_combination], 3)

        conv5 = tf.contrib.layers.conv2d(inputs=concatenation, num_outputs=5, kernel_size=[5, 5],
                                         stride=[1, 1], padding='SAME', 
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params =normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool5 = tf.contrib.layers.max_pool2d(inputs=conv5, kernel_size=[2, 2], stride=1,
                                             padding='SAME') #pooling e upsampling

        conv6 = tf.contrib.layers.conv2d(inputs=pool4, num_outputs=10, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        pool6 = tf.contrib.layers.max_pool2d(inputs=conv6, kernel_size=[2, 2], stride=1,
                                             padding='SAME')  # pooling e upsampling

        linear_combination2 = tf.contrib.layers.conv2d(inputs=pool6, num_outputs=1,
                                                       kernel_size=[1, 1], stride=[1, 1],
                                                       padding='SAME',
                                                       normalizer_fn=tf.contrib.layers.batch_norm,
                                                       normalizer_params=normalizer_params,
                                                       activation_fn=tf.nn.sigmoid)

        tf.summary.image("architecture_output", linear_combination2)
        return linear_combination2



    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
