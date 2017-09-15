import tensorflow as tf
import numpy as np
arr = np.array([[[[21], [2], [3], [4]],
                 [[6], [7], [8], [9]],
                 [[11], [12], [13], [14]],
                 [[16], [17], [18], [19]]]])
print(arr.shape)
tf_placeholder = tf.placeholder(tf.float32,shape=(arr.shape))
op1 = tf.contrib.layers.max_pool2d(inputs=tf_placeholder, kernel_size=[2, 2], stride=[2,2],
                                   padding='VALID')
op2 = tf.image.resize_nearest_neighbor(op1, [arr.shape[1], arr.shape[2]])
sess = tf.Session()
feed_dict = {tf_placeholder: arr}
print(sess.run(op2, feed_dict=feed_dict))