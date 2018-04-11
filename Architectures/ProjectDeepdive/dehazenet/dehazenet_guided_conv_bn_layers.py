import architecture
import tensorflow as tf
import Architectures.Layers.guided_conv_bn_layers as gcbl

class DehazenetGuidedConvBnLayers(architecture.Architecture):
    """
        Architecture based on https://arxiv.org/pdf/1601.07661.pdf
    """
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False):
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        # Feature extraction

        conv1 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=16, kernel_size=[5, 5],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        # Maxout 4 x 1 x 1
        pool1_1 = tf.reduce_max(conv1[:, :, :, 0:4], axis=3, keep_dims=True)
        pool1_2 = tf.reduce_max(conv1[:, :, :, 4:8], axis=3, keep_dims=True)
        pool1_3 = tf.reduce_max(conv1[:, :, :, 8:12], axis=3, keep_dims=True)
        pool1_4 = tf.reduce_max(conv1[:, :, :, 12:16], axis=3, keep_dims=True)
        pool1 = tf.concat([pool1_1, pool1_2, pool1_3, pool1_4], axis=3)

        print(pool1.shape)

        #Multi-scale mapping

        conv2_3_3 = tf.contrib.layers.conv2d(inputs=pool1, num_outputs=16, kernel_size=[3, 3],
                                             stride=[1, 1], padding='SAME',
                                             normalizer_fn=tf.contrib.layers.batch_norm,
                                             normalizer_params=normalizer_params,
                                             activation_fn=None)

        conv2_5_5 = tf.contrib.layers.conv2d(inputs=pool1, num_outputs=16, kernel_size=[5, 5],
                                             stride=[1, 1], padding='SAME',
                                             normalizer_fn=tf.contrib.layers.batch_norm,
                                             normalizer_params=normalizer_params,
                                             activation_fn=None)

        conv2_7_7 = tf.contrib.layers.conv2d(inputs=pool1, num_outputs=16, kernel_size=[7, 7],
                                             stride=[1, 1], padding='SAME',
                                             normalizer_fn=tf.contrib.layers.batch_norm,
                                             normalizer_params=normalizer_params,
                                             activation_fn=None)

        conv2 = tf.concat([conv2_3_3, conv2_5_5, conv2_7_7], axis=3)

        # Local Extremum

        max_pool = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=[7, 7], stride=1,
                                                padding='SAME') #pooling

        # Non-linear Regression

        conv3 = tf.contrib.layers.conv2d(inputs=max_pool, num_outputs=1, kernel_size=[6, 6],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        const_1 = tf.constant(1, dtype=tf.float32)
        brelu = tf.minimum(const_1, tf.nn.relu(conv3))

        guided_trans = gcbl.guidedfilter_conv_bn_layers(guided_inputs=brelu, guide_inputs=sample,
                                                        num_outputs=1, guide_kernel_size=[11, 11],
                                                        guided_kernel_size=[11, 11], stride=[1, 1],
                                                        normalizer_params=normalizer_params)
        tf.summary.image("architecture_output", guided_trans)
        return guided_trans



    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
