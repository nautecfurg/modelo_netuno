import architecture
import tensorflow as tf
import Architectures.Layers.inception_resnet_a as ira
import Architectures.Layers.inception_resnet_b as irb
import Architectures.Layers.inception_resnet_c as irc
import Architectures.Layers.guidedfilter_color_trainable_test as gct

class SibigrapiDoubleGuidedExtended2(architecture.Architecture):
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False):
        " Coarse-scale Network"
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        scale2 = [int(self.input_size[0]/4), int(self.input_size[1]/4)]
        conv1 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=16, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=16, kernel_size=[9, 9],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
        
        pool2 = tf.contrib.layers.max_pool2d(conv2, [4, 4], stride=4)

        conv1_scale2 = tf.contrib.layers.conv2d(inputs=pool2, num_outputs=16, kernel_size=[9, 9],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
        
        pool1_scale2 = tf.contrib.layers.max_pool2d(conv1_scale2, [4, 4], stride=4)
        

        upsamp1_scale2 = tf.image.resize_nearest_neighbor(pool1_scale2, scale2)    # upsampling

        conv2_scale2 = tf.contrib.layers.conv2d(inputs=upsamp1_scale2, num_outputs=16, kernel_size=[9, 9],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
        
        pool2_scale2 = tf.contrib.layers.max_pool2d(conv2_scale2, [4, 4], stride=4)
        

        upsamp2_scale2 = tf.image.resize_nearest_neighbor(pool2_scale2, scale2)    # upsampling

        upsamp2_scale2_plus_pool2 = pool2 + upsamp2_scale2

        conv3 = tf.contrib.layers.conv2d(inputs=upsamp2_scale2_plus_pool2, num_outputs=16,
                                         kernel_size=[9, 9], stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        
        pool3 = tf.contrib.layers.max_pool2d(conv3, [4, 4], stride=4)

        upsamp3 = tf.image.resize_nearest_neighbor(pool3, self.input_size)    # upsampling

        upsamp3_plus_input = upsamp3 + conv1





        module_a = ira.inception_resnet_a(upsamp3_plus_input, normalizer_params)
        module_b = irb.inception_resnet_b(module_a, normalizer_params)
        module_c = irc.inception_resnet_c(module_b, normalizer_params)

        conv4_1 = tf.contrib.layers.conv2d(inputs=module_c, num_outputs=3, kernel_size=[3, 3],
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
                guided4_1_layer = gct.guidedfilter_color_treinable(sample, conv4_1_layer, r=20, eps=10**-3)
                guided4_1_list.append(guided4_1_layer)

        guided4_1 = tf.concat(guided4_1_list, 3)

        conv4_2 = tf.contrib.layers.conv2d(inputs=module_c, num_outputs=3, kernel_size=[3, 3],
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
                guided4_2_layer = gct.guidedfilter_color_treinable(sample, conv4_2_layer, r=20, eps=10**-3)
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
