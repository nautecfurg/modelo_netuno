import tensorflow as tf
def guidedfilter(I, p, r, eps):
    """
    Summary:
        This function is a implementation of guided filter for tensorflow.
    Description:

    Args:
        I: guidance tensor (should be a gray-scale/single channel tensor).
        p: filtering input tensor (should be a gray-scale/single channel tensor).
        r: local window radius. The kernel size will be (2*r+1)x(2*r+1).
        eps: regularization parameter.

    Returns:
        q: filtering output tensor (gray-scale/single channel tensor)

    """

    #weight_conv = deep_dive.weight_variable_scaling([17, 17, 64, 64], name='weight_conv'+base_name)
    weight_conv = tf.constant(1.0, dtype=tf.float32, shape=[2*r+1, 2*r+1, 1, 1])
    ones = tf.ones(p.shape)
    n = tf.nn.conv2d(ones, weight_conv, strides=[1, 1, 1, 1], padding='SAME')
    mean_i = tf.nn.conv2d(I, weight_conv, strides=[1, 1, 1, 1], padding='SAME')/n
    mean_p = tf.nn.conv2d(p, weight_conv, strides=[1, 1, 1, 1], padding='SAME')/n
    mean_ip = tf.nn.conv2d(I*p, weight_conv, strides=[1, 1, 1, 1], padding='SAME')/n
    cov_ip = mean_ip - mean_i*mean_p # this is the covariance of (I, p) in each local patch.
    mean_ii = tf.nn.conv2d(I*I, weight_conv, strides=[1, 1, 1, 1], padding='SAME')/n
    var_i = mean_ii - mean_i*mean_i
    a = cov_ip/(var_i + eps) # Eqn. (5) in the paper
    b = mean_p - a*mean_i # Eqn. (6) in the paper
    mean_a = tf.nn.conv2d(a, weight_conv, strides=[1, 1, 1, 1], padding='SAME')/n
    mean_b = tf.nn.conv2d(b, weight_conv, strides=[1, 1, 1, 1], padding='SAME')/n

    q = mean_a*I + mean_b # Eqn. (8) in the paper
    return q
