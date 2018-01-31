import tensorflow as tf
def guidedfilter_color(I, p, r, eps):
    """
    Summary:
        This function is a implementation of guided filter for tensorflow.
    Description:

    Args:
        I: guidance tensor (should be a rgb tensor).
        p: filtering input tensor (should be a rgb tensor).
        r: local window radius. The kernel size will be (2*r+1)x(2*r+1).
        eps: regularization parameter.

    Returns:
        q: filtering output tensor (rgb tensor)

    """

    p_shape = p.get_shape().as_list()

    w_conv = tf.constant(1.0, dtype=tf.float32, shape=[2*r+1, 2*r+1, 1, 1])
    w_conv3d = tf.constant(1.0, dtype=tf.float32, shape=[2*r+1, 2*r+1, 3, 1])
    w_conv9d = tf.constant(1.0, dtype=tf.float32, shape=[2*r+1, 2*r+1, 9, 1])

    ones = tf.ones([1]+p_shape[1:])  #gambiarra para fazer o guided em um batch variavel
    N = tf.nn.conv2d(ones, w_conv, strides=[1, 1, 1, 1], padding='SAME')

    mean_I = tf.nn.depthwise_conv2d(I, w_conv3d, strides=[1, 1, 1, 1], padding='SAME')/N 
    mean_p = tf.nn.conv2d(p, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N
    mean_Ip = tf.nn.depthwise_conv2d(I*p, w_conv3d, strides=[1, 1, 1, 1], padding='SAME')/N
    cov_Ip = mean_Ip - mean_I*mean_p 
    cov_Ip = tf.expand_dims(cov_Ip, axis=-1)

    # variance of I in each local patch: the matrix Sigma in Eqn (14).
    # Note the variance in each local patch is a 3x3 symmetric matrix:
    #           rr, rg, rb
    #   Sigma = rg, gg, gb
    #           rb, gb, bb
    
    mean_I_expand = tf.expand_dims(mean_I, -1)
    mean_II = tf.matmul(mean_I_expand, mean_I_expand, transpose_a=False, transpose_b=True)

    I_expand = tf.expand_dims(I, -1)
    II = tf.matmul(I_expand, I_expand, transpose_a=False, transpose_b=True)
    II_reshaped = tf.reshape(II, [-1] + p.get_shape().as_list()[1:3]+[9])
    var_I_reshaped = tf.nn.depthwise_conv2d(II_reshaped, w_conv9d, strides=[1, 1, 1, 1], 
                                            padding='SAME')/N
    var_I = tf.reshape(var_I_reshaped, p.get_shape().as_list()[1:3]+[3, 3])
    var_I = var_I - mean_II

    sigma = var_I + (tf.eye(3,batch_shape=p.get_shape()[1:3]) * eps)

    sigma_inv = tf.matrix_inverse(sigma)

    a = tf.matmul(sigma_inv, cov_Ip, transpose_a=True)
    
    a = tf.squeeze(a, 4)        #reduce to rank 4 tensor

    b = mean_p - tf.reduce_sum(a*mean_I, axis=3, keep_dims=True)

    aI = I*tf.nn.depthwise_conv2d(a, w_conv3d, strides=[1, 1, 1, 1], padding='SAME')
    
    q = (tf.reduce_sum(aI,axis=3,keep_dims=True) \
        + tf.nn.conv2d(b, w_conv, strides=[1, 1, 1, 1], padding='SAME'))/N
    return q
