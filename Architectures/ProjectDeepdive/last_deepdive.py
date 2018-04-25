import architecture
import tensorflow as tf
import Architectures.Layers.faster_guided_filter as gf

class LastDeepDive(architecture.Architecture):
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False):
        " Coarse-scale Network"
        subsampling_ratio = 4
        lr_shape = [int(224/subsampling_ratio), int(224/subsampling_ratio)]
        lr_sample = tf.image.resize_images(sample, lr_shape)
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        nc = 16
        conv1 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=nc, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=None,
                                         activation_fn=tf.nn.relu)

        encod1 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=2*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encod1)

        encod2 = tf.contrib.layers.conv2d(inputs=encod1, num_outputs=4*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encod2)                                 
        encod3 = tf.contrib.layers.conv2d(inputs=encod2, num_outputs=8*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encod3)
        encod4 = tf.contrib.layers.conv2d(inputs=encod3, num_outputs=8*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encod4)
        
        decode1 = tf.contrib.layers.conv2d_transpose(encod4, num_outputs=8*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)

        decode1 = tf.concat([encod3, decode1], 3)

        print(decode1)

        decode2 = tf.contrib.layers.conv2d_transpose(decode1, num_outputs=4*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        decode2 = tf.concat([encod2, decode2], 3)

        print(decode2)

        decode3 = tf.contrib.layers.conv2d_transpose(decode2, num_outputs=2*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        decode3 = tf.concat([encod1, decode3], 3)

        print(decode3)

        decode4 = tf.contrib.layers.conv2d_transpose(decode3, num_outputs=nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        decode4 = tf.concat([conv1, decode4], 3)
        
        print(decode4)

        # module_a = ira.inception_resnet_a(decode4, normalizer_params)
        # module_b = irb.inception_resnet_b(module_a, normalizer_params)
        # module_c = irc.inception_resnet_c(module_b, normalizer_params)

        conv4_1 = tf.contrib.layers.conv2d(inputs=decode4, num_outputs=3, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
        lr_conv4_1 = tf.image.resize_images(conv4_1, lr_shape)
        guided4_1 = gf.fast_guided_filter(lr_sample, lr_conv4_1, sample,
                                          r=20, eps=10**-4, nhwc=True)

        conv4_2 = tf.contrib.layers.conv2d(inputs=decode4, num_outputs=3, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        lr_conv4_2 = tf.image.resize_images(conv4_2, lr_shape)
        guided4_2 = gf.fast_guided_filter(lr_sample, lr_conv4_2, sample,
                                          r=20, eps=10**-4, nhwc=True)
        
        guided4 = tf.concat([guided4_2*sample,guided4_1,sample],3)

        conv5 = tf.contrib.layers.conv2d(inputs=guided4, num_outputs=3, kernel_size=[1, 1],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=None,
                                         activation_fn=tf.nn.relu)

        brelu = tf.minimum(conv5,1)

        tf.summary.image("architecture_output", brelu)
        return brelu



    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
