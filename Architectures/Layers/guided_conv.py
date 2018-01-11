import tensorflow as tf
import numpy as np
def guidedfilter_conv2d(guided_inputs, guide_inputs, num_outputs, guide_kernel_size,
                        guided_kernel_size, stride, padding='SAME',):
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
                                    activation_fn=None)

    strides = [1]+stride+[1]
    ksizes = [1] + guided_kernel_size + [1]
    extracted_image = tf.extract_image_patches(guided_inputs, ksizes=ksizes, strides=strides,
                                               rates=[1,1,1,1], padding=padding)
    print(extracted_image)
    print(weights)
    extracted_image_resized = tf.concat([extracted_image] * num_outputs, 3)
    print(extracted_image_resized)

    prod_image_weights =  extracted_image_resized * weights

    ones = tf.ones(shape=[1, 1, kernel_length, num_outputs])
    const = tf.constant(0.1, shape=[num_outputs])
    b_conv = tf.Variable(const)
    
    outputs = tf.nn.conv2d(prod_image_weights, ones,
                           strides=strides, padding='SAME') + b_conv

    return outputs
