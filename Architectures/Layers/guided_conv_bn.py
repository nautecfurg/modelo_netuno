import tensorflow as tf
import numpy as np


def sum_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_sum(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def guidedfilter_conv_bn(guided_inputs, guide_inputs, num_outputs, guide_kernel_size,
                        guided_kernel_size, stride, normalizer_params, activation_fn, padding='SAME'):
    """
    Summary:
        This function performs what we call guided convolution
    Description:
        The `guide_inputs` is used to generate a kernel to each pixel for the `guided_inputs`, 
        every kernel has `guided_kernel_size` and is generated by a `guide_kernel_size`. 
        These kernels are convolved in the `guided_inputs` generating a feature map with
        `num_outputs` channels
    Args:
        

    Returns:
        

    """

    guided_inputs_depth = int(guided_inputs.shape[3])
    guided_kernel_prod = np.prod(guided_kernel_size)
    kernel_length = guided_kernel_prod*num_outputs*guided_inputs_depth

    weights = tf.contrib.layers.conv2d(inputs=guide_inputs, num_outputs=kernel_length,
                                    kernel_size=guide_kernel_size,
                                    stride=stride, padding=padding,
                                    normalizer_fn=None,
                                    normalizer_params=None,
                                    activation_fn=None)

    strides = [1]+stride+[1]
    ksizes = [1] + guided_kernel_size + [1]

    extracted_image = tf.extract_image_patches(guided_inputs, ksizes=ksizes, strides=strides,
                                               rates=[1,1,1,1], padding=padding)
    extracted_image_resized = tf.concat([extracted_image] * num_outputs, 3)

    prod_image_weights =  extracted_image_resized * weights

    const = tf.constant(0.1, shape=[num_outputs])
    b_conv = tf.Variable(const)
    outputs = activation_fn(sum_out(inputs=prod_image_weights, num_units=num_outputs) + b_conv)


    return outputs
