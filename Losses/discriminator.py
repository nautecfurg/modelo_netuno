import numpy as np
import tensorflow as tf

import loss
import optimizer

class DiscriminatorLoss(loss.Loss):
    """This class is responsible for creating a Generative Adversarial Network and calculate
    the network loss.

    The class enables the usage of GANs on the project which are trainable and can evaluate
    the loss for given networks.
    """

    def __init__(self):
        """This constructor initializes a new DiscriminatorLoss object.

        The function loads the parameters from the discriminator json (to be defined) and
        creates a new object based on them.

        Returns:
            Nothing.
        """
        parameters_list = []
        self.config_dict = self.open_config(parameters_list)

        # Define defaults
        self.disc_gt = 0.0
        self.disc_out = 0.0

    def evaluate(self, architecture_output, target_output):
        """This method evaluates the loss for the given image and it's ground-truth.

        The method models a discriminator neural network on a separate variable scope
        and allows for the calculation of the loss.

        Args:
            architecture_output: The image to input in the discriminator network.

            target_output: The ground-truth image to input in the discriminator network.

        Returns:
            The value of the discriminator loss returned by the network.
        """

        def discriminator_layer(image, name, n, depth, stride, training=True):
            """This function creates one layer of the discriminator network.

            This function is to be called when creating the structure of the
            discriminator network as it's often used.

            Args:
                image: The image to input in the convolutions.

                name: The name of the layer.

                n: the fourth dimension of the shape of the weights.

                depth: the third dimension of the shape of the weights.

                stride: the stride to use in the convolution.

            Returns:
                The resulting activations after applying the layer.
            """
            weights = tf.get_variable(shape=[3, 3, depth, n], name="weights" + name,
                                      initializer=tf.uniform_unit_scaling_initializer(factor=0.01))
            #biases = tf.Variable(tf.constant(0.01, shape=[n]), name="biases" + name)
            biases = tf.get_variable(shape=[n], name="biases" + name, 
                                      initializer=tf.constant_initializer(value=0.01))

            conv = tf.nn.conv2d(image, weights, strides=[1, stride, stride, 1],
                                padding="VALID") + biases
            leaky = tf.maximum(0.1 * conv, conv)

            return tf.contrib.layers.batch_norm(leaky, center=True, updates_collections=None,
                                                scale=True, is_training=training)

        def discriminator_network(image):
            # Input Layer
            weights = tf.get_variable(shape=[3, 3, 3, 64], name="weights1",
                                      initializer=tf.uniform_unit_scaling_initializer(factor=0.01))
            #biases = tf.Variable(tf.constant(0.01, shape=[64]), name="biases1")
            biases = tf.get_variable(shape=[64], name="biases1", 
                                      initializer=tf.constant_initializer(value=0.01))
            conv = tf.nn.conv2d(image, weights, strides=[1, 1, 1, 1],
                                padding="SAME") + biases
            leaky = tf.maximum(0.1 * conv, conv)

            # Discriminator Layers
            layer1 = discriminator_layer(leaky, "A", 64, 64, 2, training=True)
            layer2 = discriminator_layer(layer1, "B", 128, 64, 1, training=True)
            layer3 = discriminator_layer(layer2, "C", 128, 128, 2, training=True)
            layer4 = discriminator_layer(layer3, "D", 256, 128, 1, training=True)
            layer5 = discriminator_layer(layer4, "E", 256, 256, 2, training=True)
            layer6 = discriminator_layer(layer5, "F", 512, 256, 2, training=True)
            layer7 = discriminator_layer(layer6, "G", 512, 512, 2, training=True)
            layer8 = discriminator_layer(layer7, "H", 512, 512, 2, training=True)

            # Output Layer
            shape = int(np.prod(layer8.get_shape()[1:]))
            flat = tf.reshape(layer8, [-1, shape])
            weights = tf.get_variable(shape=[shape, 1], name="weights2", dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1e-1))
            biases = tf.get_variable(shape=[1], name="biases2", dtype=tf.float32,
                                     initializer=tf.constant_initializer(1.0))
            connect = tf.matmul(flat, weights) + biases

            return tf.maximum(0.1 * connect, connect)

        with tf.variable_scope("discriminator", reuse=None):
            self.disc_gt = discriminator_network(target_output)

        with tf.variable_scope("discriminator", reuse=True):
            self.disc_out = discriminator_network(architecture_output)

        # Network Loss
        #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.disc_out, labels=tf.ones_like(self.disc_out)))
        loss = tf.reduce_mean(-tf.log(self.disc_out + 1e-12))

        return loss

    def train(self, optimizer_imp):
        """This method returns the training operation of the network.

        This method returns the training operation that is to be runned by tensorflow
        to minimize the discriminator network in relation to it's own error.

        Args:
            optimizer_imp: The implementation of the optimizer to use.

        Returns:
            The operation to run to optimize the discriminator network.
        """
        #disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/discriminator")
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/discriminator")

        # Discriminator Loss
        #disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.disc_gt, labels=tf.ones_like(self.disc_gt)), name="disc_real_loss")
        #disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.disc_out, labels=tf.zeros_like(self.disc_out)), name="disc_fake_loss")
        #disc_loss = tf.add(disc_fake, disc_real)
        disc_fake = tf.log(1 - self.disc_out + 1e-12)
        disc_real = tf.log(self.disc_gt + 1e-12)
        disc_loss = tf.reduce_mean(-(disc_fake + disc_real))

        # Add To Summary
        tf.summary.scalar("discriminator_loss_real", disc_real)
        tf.summary.scalar("discriminator_loss_fake", disc_fake)
        tf.summary.scalar("discriminator_loss", disc_loss)

        # Optimization
        disc_train = optimizer_imp.minimize(disc_loss, var_list=disc_vars)
        return disc_train

    def trainable(self):
        """This method tells whether this network is trainable or not.

        This method overrides the parent default method to make this network be trained on
        the main loop of the project.

        Returns:
            True, as the network is trainable.
        """
        return True
