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

    eps_tensor = tf.constant(eps)
    depth = I.shape[3]
    mean_I = tf.nn.pool(input=I,
        window_shape=[2*r+1, 2*r+1],
        pooling_type="AVG",
        padding="SAME")
    mean_p = tf.nn.pool(input=p,
        window_shape=[2*r+1, 2*r+1],
        pooling_type="AVG",
        padding="SAME")
    mean_Ip = tf.nn.pool(input=tf.multiply(I, p),
        window_shape=[2*r+1, 2*r+1],
        pooling_type="AVG",
        padding="SAME")
    
    cov_Ip = tf.subtract(mean_Ip, tf.multiply(mean_I, mean_p))

    mean_II = tf.nn.pool(input=tf.multiply(I, I),
        window_shape=[2*r+1, 2*r+1],
        pooling_type="AVG",
        padding="SAME")
    
    var_I = tf.subtract(mean_II, tf.multiply(mean_I, mean_I))

    a = tf.divide(cov_Ip, var_I + eps_tensor)
    b = tf.subtract(mean_p, tf.multiply(a, mean_I))

    mean_a = tf.nn.pool(input=a,
        window_shape=[2*r+1, 2*r+1],
        pooling_type="AVG",
        padding="SAME")
    mean_b = tf.nn.pool(input=b,
        window_shape=[2*r+1, 2*r+1],
        pooling_type="AVG",
        padding="SAME")

    q = tf.multiply(mean_a, tf.add(I, mean_b))
    