import loss
import tensorflow as tf
class SSIM(loss.Loss):
    def __init__(self):
        parameters_list = []
        self.config_dict = self.open_config(parameters_list)
    def evaluate(self, architecture_output, target_output):

        C1 = 6.5025
        C2 = 58.5225

        image1 = architecture_output
        image2 = target_output

        image1_1 = image1 * image1 # square the image
        image2_2 = image2 * image2
        image1_2 = image1 * image2
        # clur the images

        #pylint: disable=line-too-long
        weights = [[[0.0000, 0.0000, 0.0000, 0.0001, 0.0002, 0.0003, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0001, 0.0003, 0.0008, 0.0016, 0.0020, 0.0016, 0.0008, 0.0003, 0.0001, 0.0000],
                    [0.0000, 0.0003, 0.0013, 0.0039, 0.0077, 0.0096, 0.0077, 0.0039, 0.0013, 0.0003, 0.0000],
                    [0.0001, 0.0008, 0.0039, 0.0120, 0.0233, 0.0291, 0.0233, 0.0120, 0.0039, 0.0008, 0.0001],
                    [0.0002, 0.0016, 0.0077, 0.0233, 0.0454, 0.0567, 0.0454, 0.0233, 0.0077, 0.0016, 0.0002],
                    [0.0003, 0.0020, 0.0096, 0.0291, 0.0567, 0.0708, 0.0567, 0.0291, 0.0096, 0.0020, 0.0003],
                    [0.0002, 0.0016, 0.0077, 0.0233, 0.0454, 0.0567, 0.0454, 0.0233, 0.0077, 0.0016, 0.0002],
                    [0.0001, 0.0008, 0.0039, 0.0120, 0.0233, 0.0291, 0.0233, 0.0120, 0.0039, 0.0008, 0.0001],
                    [0.0000, 0.0003, 0.0013, 0.0039, 0.0077, 0.0096, 0.0077, 0.0039, 0.0013, 0.0003, 0.0000],
                    [0.0000, 0.0001, 0.0003, 0.0008, 0.0016, 0.0020, 0.0016, 0.0008, 0.0003, 0.0001, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0001, 0.0002, 0.0003, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000]]]
        #pylint: enable=line-too-long
        weights = tf.concat(0, [weights] * 3)
        weights = tf.reshape(weights, [11, 11, 3, 1])

        # create a typical 11x11 gausian kernel with 1.5 sigma
        conv1 = tf.nn.conv2d(image1, weights, strides=[1, 1, 1, 1], padding='VALID')
        conv2 = tf.nn.conv2d(image2, weights, strides=[1, 1, 1, 1], padding='VALID')

        conv1_1 = conv1 * conv1 # square the image
        conv2_2 = conv2 * conv2
        conv1_2 = conv1 * conv2

        sigma1_1 = tf.nn.conv2d(image1_1, weights, strides=[1, 1, 1, 1], padding='VALID')
        sigma1_1 = sigma1_1 - conv1_1


        sigma2_2 = tf.nn.conv2d(image2_2, weights, strides=[1, 1, 1, 1], padding='VALID')
        sigma2_2 = sigma2_2 - conv2_2


        sigma1_2 = tf.nn.conv2d(image1_2, weights, strides=[1, 1, 1, 1], padding='VALID')
        sigma1_2 = sigma1_2 - conv1_2

        temp1 = conv1_2 * 2
        temp1 = temp1 + C1
        temp2 = sigma1_2 * 2
        temp2 = temp2 + C2
        temp3 = temp1 * temp2

        temp1 = conv1_1 + conv2_2
        temp1 = temp1 + C1
        temp2 = sigma1_1 + sigma2_2
        temp2 = temp2 + C2

        temp1 = temp1 * temp2

        ssim_map = temp3 / temp1
        ssim = tf.reduce_mean(ssim_map)
        ssim = 1.0 - ssim
        return ssim
