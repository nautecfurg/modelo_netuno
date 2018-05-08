import architecture
import tensorflow as tf

class Deeplabv2(architecture.Architecture):
    """
        Deeplabv2 https://arxiv.org/pdf/1606.00915.pdf
    """
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                "validation_period", "model_saving_period"]
        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False, num_classes=2):
        '''Network definition.

        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''

        """
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))
        """
        normalizer_params = {'is_training':training, 'center':True,
                'updates_collections':None, 'scale':True}

        conv1 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=64, kernel_size=[7, 7],
                stride=[2, 2], padding='SAME',
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=normalizer_params)
        pool1 = tf.contrib.layers.max_pool2d(inputs=conv1, kernel_size=[3, 3], stride=[2, 2],
                padding='SAME')
        res2a_branch1 = tf.contrib.layers.conv2d(inputs=pool1, num_outputs=256, kernel_size=[1, 1],
                stride=[1, 1], padding='SAME',
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=normalizer_params,
                activation_fn=None)

        """ 
        (self.feed('pool1')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))
        """
        res2a_branch2a = tf.contrib.layers.conv2d(inputs=pool1, num_outputs=64, kernel_size=[1, 1],
                stride=[1, 1], padding='SAME',
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=normalizer_params)
        res2a_branch2b = tf.contrib.layers.conv2d(inputs=res2a_branch2a, num_outputs=64, kernel_size=[3, 3],
                stride=[1, 1], padding='SAME',
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=normalizer_params)
        res2a_branch2c = tf.contrib.layers.conv2d(inputs=res2a_branch2b, num_outputs=256, kernel_size=[1, 1],
                stride=[1, 1], padding='SAME',
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=normalizer_params,
                activation_fn=None)
        """
        (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
             .add(name='res2a')
             .relu(name='res2a_relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
             .batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
             .batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))
        """
        res2a = tf.add_n([res2a_branch1, res2a_branch2c])
        res2a_relu = tf.nn.relu(res2a)

        res2b_branch2a = tf.contrib.layers.conv2d(inputs=res2a_relu, num_outputs=64, kernel_size=[1, 1],
                stride=[1, 1], padding='SAME',
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=normalizer_params)
        res2b_branch2b = tf.contrib.layers.conv2d(inputs=res2b_branch2a, num_outputs=64, kernel_size=[3, 3],
                stride=[1, 1], padding='SAME',
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=normalizer_params)
        res2b_branch2c = tf.contrib.layers.conv2d(inputs=res2b_branch2b, num_outputs=256, kernel_size=[1, 1],
                stride=[1, 1], padding='SAME',
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=normalizer_params,
                activation_fn=None)


