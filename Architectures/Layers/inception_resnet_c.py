import tensorflow as tf

def inception_resnet_c(input_layer, normalizer_params):
    
    conv1 = tf.contrib.layers.conv2d(inputs=input_layer, num_outputs=32, kernel_size=[1, 1],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
    
    conv2 = tf.contrib.layers.conv2d(inputs=input_layer, num_outputs=32, kernel_size=[1, 1],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
    
    conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=32, kernel_size=[1, 7],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

    conv4 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=32, kernel_size=[7, 1],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
    
    concatenation = tf.concat([conv1, conv4], 3)


    conv5 = tf.contrib.layers.conv2d(inputs=concatenation, num_outputs=16, kernel_size=[1, 1],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)
    
    result = tf.add(conv5, input_layer)

    return result