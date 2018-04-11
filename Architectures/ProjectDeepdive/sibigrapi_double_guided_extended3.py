import architecture
import tensorflow as tf
import Architectures.Layers.inception_resnet_a as ira
import Architectures.Layers.inception_resnet_b as irb
import Architectures.Layers.inception_resnet_c as irc
import Architectures.Layers.guidedfilter_color_trainable as gct

class SibigrapiDoubleGuidedExtended3(architecture.Architecture):
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False):
        " Coarse-scale Network"
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
        

        guided4_1_list = []
        for i in range(3):
            if i == 0:
                reuse = None
            else:
                reuse = True
            
            with tf.variable_scope("guided",reuse=reuse):
                conv4_1_layer =tf.expand_dims(conv4_1[:,:,:,i], -1) 
<<<<<<< HEAD
                guided4_1_layer = gct.guidedfilter_color_treinable(sample, conv4_1_layer, r=20, eps=10**-4)
=======
                guided4_1_layer = gct.guidedfilter_color_treinable(sample, conv4_1_layer, r=20, eps=10**-3)
>>>>>>> eba5fadb58cb5b0d37a68d7b6c68d9a547ad421b
                guided4_1_list.append(guided4_1_layer)

        guided4_1 = tf.concat(guided4_1_list, 3)

        conv4_2 = tf.contrib.layers.conv2d(inputs=decode4, num_outputs=3, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        guided4_2_list = []
        for i in range(3):
            if i == 0:
                reuse = None
            else:
                reuse = True
            
            with tf.variable_scope("guided2",reuse=reuse):
                conv4_2_layer =tf.expand_dims(conv4_2[:,:,:,i], -1) 
<<<<<<< HEAD
                guided4_2_layer = gct.guidedfilter_color_treinable(sample, conv4_2_layer, r=20, eps=10**-4)
=======
                guided4_2_layer = gct.guidedfilter_color_treinable(sample, conv4_2_layer, r=20, eps=10**-3)
>>>>>>> eba5fadb58cb5b0d37a68d7b6c68d9a547ad421b
                guided4_2_list.append(guided4_2_layer)

        guided4_2 = tf.concat(guided4_2_list, 3)

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
