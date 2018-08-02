import architecture
import tensorflow as tf
from Architectures.Layers.choquet_pool import *

class DehazenetChoquet(architecture.Architecture):
    """
        Architecture based on https://arxiv.org/pdf/1601.07661.pdf
    """
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]
        self.layers_dict = {}

    def prediction(self, sample, training=False):
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        # Feature extraction
        #sample=tf.image.crop_to_bounding_box(sample, 0, 0, 16, 16)
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
        self.layers_dict["conv2"]=conv2

        conv2a = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=1, kernel_size=[1, 1],
                                             stride=[1, 1], padding='SAME',
                                             normalizer_fn=tf.contrib.layers.batch_norm,
                                             normalizer_params=normalizer_params,
                                             activation_fn=None)
        self.layers_dict["conv2a"]=conv2a

        # Local Extremum

        max_pool = tf.contrib.layers.max_pool2d(inputs=conv2, kernel_size=[7, 7], stride=1,
                                                padding='SAME') #pooling
        self.layers_dict["max_pool"]=max_pool

        #choq_pool = choquet_pooling(conv2a, [1,7,7,1], [1,1,1,1], [1,1,1,1], 'SAME', MF)
        choq_pool = fast_choquet_pooling_trainable(tf.nn.relu(conv2a), [1,7,7,1], [1,1,1,1], [1,1,1,1], 'SAME')
        self.layers_dict["choq_pool"]=choq_pool

        pool2 = tf.concat([max_pool, choq_pool], axis=3)

        # Non-linear Regression

        conv3 = tf.contrib.layers.conv2d(inputs=pool2, num_outputs=1, kernel_size=[6, 6],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        const_1 = tf.constant(1, dtype=tf.float32)
        brelu = tf.minimum(const_1, tf.nn.relu(conv3))
        tf.summary.image("architecture_output", brelu)
        return brelu



    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]

    def get_layer(self, layer_name):
        layer=self.layers_dict[layer_name]
        return layer
