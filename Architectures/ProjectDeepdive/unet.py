import architecture
import tensorflow as tf

def leaky_relu(x, alpha=0.2):
      return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

class Unet(architecture.Architecture):
    

    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False):
        
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        ngf = 64
        paddings = tf.constant([[0,0], [1, 1,], [1, 1], [0, 0]])
        # input is (nc) x 256 x 256
        
        sample = tf.image.resize_images(sample, size=[256, 256]) 
        
        sample_pad = tf.pad(sample, paddings)
        print(sample)
        print(sample_pad)
        encode1 = tf.contrib.layers.conv2d(inputs=sample_pad, num_outputs=ngf, kernel_size=[4, 4],
                                         stride=[2, 2], padding='VALID',
                                         normalizer_fn=None,
                                         activation_fn=None)
        print(encode1)
        # input is (ngf) x 128 x 128
        conv1 = leaky_relu(encode1)
        conv1_pad = tf.pad(conv1, paddings)
        encode2 = tf.contrib.layers.conv2d(inputs=conv1_pad, num_outputs=2*ngf, kernel_size=[4, 4],
                                         stride=[2, 2], padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)        
        print(encode2)
        # input is (ngf * 2) x 64 x 64
        conv2 = leaky_relu(encode2)
        conv2_pad = tf.pad(conv2, paddings)
        encode3 = tf.contrib.layers.conv2d(inputs=conv2_pad, num_outputs=4*ngf, kernel_size=[4, 4],
                                         stride=[2, 2], padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        print(encode3)
        # input is (ngf * 4) x 32 x 32
        conv3 = leaky_relu(encode3)
        conv3_pad = tf.pad(conv3, paddings)
        encode4 = tf.contrib.layers.conv2d(inputs=conv3_pad, num_outputs=8*ngf, kernel_size=[4, 4],
                                         stride=[2, 2], padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        print(encode4)
        # input is (ngf * 8) x 16 x 16
        conv4 = leaky_relu(encode4)
        conv4_pad = tf.pad(conv4, paddings)
        encode5 = tf.contrib.layers.conv2d(inputs=conv4_pad, num_outputs=8*ngf, kernel_size=[4, 4],
                                         stride=[2, 2], padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        print(encode5)
        # input is (ngf * 8) x 8 x 8
        conv5 = leaky_relu(encode5)
        conv5_pad = tf.pad(conv5, paddings)
        encode6 = tf.contrib.layers.conv2d(inputs=conv5_pad, num_outputs=8*ngf, kernel_size=[4, 4],
                                         stride=[2, 2], padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        print(encode6)
        # input is (ngf * 8) x 4 x 4
        conv6 = leaky_relu(encode6)
        conv6_pad = tf.pad(conv6, paddings)
        encode7 = tf.contrib.layers.conv2d(inputs=conv6_pad, num_outputs=8*ngf, kernel_size=[4, 4],
                                         stride=[2, 2], padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=None)
        print(encode7)
        # input is (ngf * 8) x 2 x 2
        conv7 = leaky_relu(encode7)
        conv7_pad = tf.pad(conv7, paddings)
        encode8 = tf.contrib.layers.conv2d(inputs=conv7_pad, num_outputs=8*ngf, kernel_size=[4, 4],
                                         stride=[2, 2], padding='VALID',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
        print(encode8)
        # input is (ngf * 8) x 1 x 1
        decode1 = tf.contrib.layers.conv2d_transpose(encode8, num_outputs=8*ngf,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=None)

#        decode1 = tf.image.crop_to_bounding_box(decode1,
#                                                offset_height = 1,
#                                                offset_width = 1,
#                                                target_height = decode1.get_shape()[1]-2,
#                                                target_width = decode1.get_shape()[2]-2)

        decode1_drop = tf.nn.dropout(decode1, keep_prob=0.5)
        print(decode1_drop)
        decode1 = tf.nn.relu(tf.concat([decode1_drop, encode7],3))

        # input is (ngf * 8) x 2 x 2
        # decode1_pad = tf.pad(decode1, paddings)
        decode2 = tf.contrib.layers.conv2d_transpose(decode1, num_outputs=8*ngf,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=None)

 #       decode2 = tf.image.crop_to_bounding_box(decode2,
 #                                               offset_height = 1,
 #                                               offset_width = 1,
 #                                               target_height = decode2.get_shape()[1]-2,
 #                                               target_width = decode2.get_shape()[2]-2)

        decode2_drop = tf.nn.dropout(decode2, keep_prob=0.5)
        decode2 = tf.nn.relu(tf.concat([decode2_drop, encode6],3))

        # input is (ngf * 8) x 4 x 4
        # decode2_pad = tf.pad(decode2, paddings)
        decode3 = tf.contrib.layers.conv2d_transpose(decode2, num_outputs=8*ngf,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=None)

#        decode3 = tf.image.crop_to_bounding_box(decode3,
#                                                offset_height = 1,
#                                                offset_width = 1,
#                                                target_height = decode3.get_shape()[1]-2,
#                                                target_width = decode3.get_shape()[2]-2)

        decode3_drop = tf.nn.dropout(decode3, keep_prob=0.5)
        decode3 = tf.nn.relu(tf.concat([decode3_drop, encode5],3))


        # input is (ngf * 8) x 8 x 8
        # decode3_pad = tf.pad(decode3, paddings)
        decode4 = tf.contrib.layers.conv2d_transpose(decode3, num_outputs=8*ngf,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=None)

#        decode4 = tf.image.crop_to_bounding_box(decode4,
#                                                offset_height = 1,
#                                                offset_width = 1,
#                                                target_height = decode4.get_shape()[1]-2,
#                                                target_width = decode4.get_shape()[2]-2)
        decode4 = tf.nn.relu(tf.concat([decode4, encode4],3))

        # input is (ngf * 8) x 16 x 16
        # decode4_pad = tf.pad(decode4, paddings)
        decode5 = tf.contrib.layers.conv2d_transpose(decode4, num_outputs=4*ngf,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=None)

#        decode5 = tf.image.crop_to_bounding_box(decode5,
#                                                offset_height = 1,
#                                                offset_width = 1,
#                                                target_height = decode5.get_shape()[1]-2,
#                                                target_width = decode5.get_shape()[2]-2)

        decode5 = tf.nn.relu(tf.concat([decode5, encode3],3))

        # input is (ngf * 4) x 32 x 32
        # decode5_pad = tf.pad(decode5, paddings)
        decode6 = tf.contrib.layers.conv2d_transpose(decode5, num_outputs=2*ngf,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=None)
#        decode6 = tf.image.crop_to_bounding_box(decode6,
#                                                offset_height = 1,
#                                                offset_width = 1,
#                                                target_height = decode6.get_shape()[1]-2,
#                                                target_width = decode6.get_shape()[2]-2)
        decode6 = tf.nn.relu(tf.concat([decode6, encode2],3))

        # input is (ngf * 2) x 64 x 64
        # decode6_pad = tf.pad(decode6, paddings)
        decode7 = tf.contrib.layers.conv2d_transpose(decode6, num_outputs=ngf,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=None)

#        decode7 = tf.image.crop_to_bounding_box(decode7,
#                                                offset_height = 1,
#                                                offset_width = 1,
#                                                target_height = decode7.get_shape()[1]-2,
#                                                target_width = decode7.get_shape()[2]-2)

        decode7 = tf.nn.relu(tf.concat([decode7, encode1],3))

        # input is (ngf) x 128 x 128
        # decode7_pad = tf.pad(decode7, paddings)
        decode8 = tf.contrib.layers.conv2d_transpose(decode7, num_outputs=3,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.tanh)

#        decode8 = tf.image.crop_to_bounding_box(decode8,
#                                                offset_height = 1,
#                                                offset_width = 1,
#                                                target_height = decode8.get_shape()[1]-2,
#                                                target_width = decode8.get_shape()[2]-2)

        decode8 = tf.image.resize_images(decode8, size=self.input_size) 

        tf.summary.image('architecture output', decode8)

        return decode8



    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
