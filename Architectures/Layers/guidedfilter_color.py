import tensorflow as tf
def guidedfilter_color(I,p,r,eps):
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


    w_conv = tf.constant(1.0, dtype=tf.float32, shape=[2*r+1, 2*r+1, 1, 1])
    ones = tf.ones(p.shape)
    N = tf.nn.conv2d(ones, w_conv, strides=[1, 1, 1, 1], padding='SAME')
    I_r = tf.expand_dims(I[:, :, :, 0], -1)
    I_g = tf.expand_dims(I[:, :, :, 1], -1)
    I_b = tf.expand_dims(I[:, :, :, 2], -1)
    mean_I_r = tf.nn.conv2d(I_r, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N
    mean_I_g = tf.nn.conv2d(I_g, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N
    mean_I_b = tf.nn.conv2d(I_b, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N

    mean_p = tf.nn.conv2d(p, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N

    mean_Ip_r = tf.nn.conv2d(I_r*p, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N
    mean_Ip_g = tf.nn.conv2d(I_g*p, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N
    mean_Ip_b = tf.nn.conv2d(I_b*p, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N

    # this is the covariance of (I, p) in each local patch.
    cov_Ip_r = mean_Ip_r - mean_I_r*mean_p
    cov_Ip_g = mean_Ip_g - mean_I_g*mean_p
    cov_Ip_b = mean_Ip_b - mean_I_b*mean_p

    # variance of I in each local patch: the matrix Sigma in Eqn (14).
    # Note the variance in each local patch is a 3x3 symmetric matrix:
    #           rr, rg, rb
    #   Sigma = rg, gg, gb
    #           rb, gb, bb
    mean_I_rr = mean_I_r * mean_I_r
    mean_I_rg = mean_I_r * mean_I_g
    mean_I_rb = mean_I_r * mean_I_b
    mean_I_gg = mean_I_g * mean_I_g
    mean_I_gb = mean_I_g * mean_I_b
    mean_I_bb = mean_I_b * mean_I_b

    var_I_rr = tf.nn.conv2d(I_r*I_r, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N - mean_I_rr
    var_I_rg = tf.nn.conv2d(I_r*I_g, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N - mean_I_rg
    var_I_rb = tf.nn.conv2d(I_r*I_b, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N - mean_I_rb
    var_I_gg = tf.nn.conv2d(I_g*I_g, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N - mean_I_gg
    var_I_gb = tf.nn.conv2d(I_g*I_b, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N - mean_I_gb
    var_I_bb = tf.nn.conv2d(I_b*I_b, w_conv, strides=[1, 1, 1, 1], padding='SAME')/N - mean_I_bb

    sigma1 = tf.concat([var_I_rr+eps, var_I_rg, var_I_rb], 3)
    sigma2 = tf.concat([var_I_rg, var_I_gg+eps, var_I_gb], 3)
    sigma3 = tf.concat([var_I_rb, var_I_gb, var_I_bb+eps], 3)

    sigma1 = tf.expand_dims(sigma1, -1)
    sigma2 = tf.expand_dims(sigma2, -1)
    sigma3 = tf.expand_dims(sigma3, -1)
    sigma = tf.concat([sigma1, sigma2, sigma3], 4)
    sigma_inv = tf.matrix_inverse(sigma)

    cov_Ip_r = tf.expand_dims(cov_Ip_r, -1)
    cov_Ip_g = tf.expand_dims(cov_Ip_g, -1)
    cov_Ip_b = tf.expand_dims(cov_Ip_b, -1)
    cov_Ip = tf.concat([cov_Ip_r, cov_Ip_g, cov_Ip_b], 4)

    a = tf.matmul(cov_Ip, sigma_inv)

    b = mean_p - a[:, :, :, :, 0] * mean_I_r - a[:, :, :, :, 1] *\
        mean_I_g - a[:, :, :, :, 2]*mean_I_b

    q = (I_r* tf.nn.conv2d(a[:, :, :, :, 0], w_conv, strides=[1, 1, 1, 1], padding='SAME') \
        + I_g * tf.nn.conv2d(a[:, :, :, :, 1], w_conv, strides=[1, 1, 1, 1], padding='SAME') \
        + I_b * tf.nn.conv2d(a[:, :, :, :, 2], w_conv, strides=[1, 1, 1, 1], padding='SAME') \
        + tf.nn.conv2d(b, w_conv, strides=[1, 1, 1, 1], padding='SAME'))/N

    return q
