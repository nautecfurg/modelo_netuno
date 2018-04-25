import architecture
import tensorflow as tf
import Architectures.Layers.faster_guided_filter as gf


class GuidedNet(architecture.Architecture):
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

        encode1 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=2*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encode1)

        encode2 = tf.contrib.layers.conv2d(inputs=encode1, num_outputs=4*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encode2)                                 
        encode3 = tf.contrib.layers.conv2d(inputs=encode2, num_outputs=8*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encode3)
        encode4 = tf.contrib.layers.conv2d(inputs=encode3, num_outputs=8*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encode4)
        
        decode1 = tf.contrib.layers.conv2d_transpose(encode4, num_outputs=8*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)

        skip1 = tf.concat([encode3, decode1], 3)

        print(skip1)

        decode2 = tf.contrib.layers.conv2d_transpose(skip1, num_outputs=4*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        skip2 = tf.concat([encode2, decode2], 3)

        print(skip2)

        decode3 = tf.contrib.layers.conv2d_transpose(skip2, num_outputs=2*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        skip3 = tf.concat([encode1, decode3], 3)

        print(skip3)

        decode4 = tf.contrib.layers.conv2d_transpose(skip3, num_outputs=nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        #guided = tf.concat([guided_filter(conv1, decode4, nc), decode4], 3)

        lr_conv1 = tf.image.resize_images(conv1, lr_shape)
        lr_decode4 = tf.image.resize_images(decode4, lr_shape)
        guided = gf.guided_filter(lr_conv1, lr_decode4, r=20, eps=10**-4)
        guided = gf.fast_guided_filter(lr_conv1, lr_decode4, conv1,
                                        r=20, eps=10**-4, nhwc=True)
        
        print(guided)

        conv4_1 = tf.contrib.layers.conv2d(inputs=guided, num_outputs=3, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
        

        #guided4_1 = guided_filter(sample, conv4_1, 3)
        #guided4_1 = gf.guided_filter(sample, conv4_1, r=20, eps=10**-4)
        lr_conv4_1 = tf.image.resize_images(conv4_1, lr_shape)
        guided4_1 = gf.fast_guided_filter(lr_sample, lr_conv4_1, sample,
                                          r=20, eps=10**-4, nhwc=True)
        conv4_2 = tf.contrib.layers.conv2d(inputs=guided, num_outputs=3, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        #guided4_2 = guided_filter(sample, conv4_2, 3)
        lr_conv4_2 = tf.image.resize_images(conv4_2, lr_shape)
        #guided4_2 = gf.guided_filter(sample, conv4_2, r=20, eps=10**-4)
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
