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
        
        def lrelu(image, alpha):
            return tf.maximum(alpha * image, image)
            
        # New Layer
        def discriminator_layer(image, output_channel, kernel_size, stride, name):
            output = tf.contrib.layers.conv2d(image, output_channel, kernel_size, stride, 'SAME', data_format='NHWC',
                                              activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              biases_initializer=None)
            output = tf.contrib.layers.batch_norm(output, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                                  scale=False, fused=True, is_training=True)
            output = lrelu(output, 0.2)
            
            return output

        # New Network
        def discriminator_network(image):
            # Input Layer
            output = tf.contrib.layers.conv2d(image, 64, 3, 1, 'SAME', data_format='NHWC',
                                              activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
            output = lrelu(output, 0.2)
            
            # Discriminator Layers
            output = discriminator_layer(output, 64, 3, 2, "A")
            output = discriminator_layer(output, 128, 3, 1, "B")
            output = discriminator_layer(output, 128, 3, 2, "C")
            output = discriminator_layer(output, 256, 3, 1, "D")
            output = discriminator_layer(output, 256, 3, 2, "E")
            output = discriminator_layer(output, 512, 3, 1, "F")
            output = discriminator_layer(output, 512, 3, 2, "G")
            
            # Dense Layer 1
            output = tf.layers.flatten(output)
            output = tf.layers.dense(output, 1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            output = lrelu(output, 0.2)
            
            # Dense Layer 2
            output = tf.layers.dense(output, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #output = tf.nn.sigmoid(output)
            
            return output, tf.nn.sigmoid(output)
            
        with tf.variable_scope("discriminator", reuse=None):
            self.disc_gt, self.disc_gt_sigmoid = discriminator_network(target_output)

        with tf.variable_scope("discriminator", reuse=True):
            self.disc_out, self.disc_out_sigmoid = discriminator_network(architecture_output)

        #adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.disc_out, labels=tf.ones_like(self.disc_out_sigmoid)))

        adv_loss = tf.reduce_mean((self.disc_out - tf.ones_like(self.disc_out_sigmoid))**2)

        return adv_loss

    def train(self, optimizer_imp):
        """This method returns the training operation of the network.

        This method returns the training operation that is to be runned by tensorflow
        to minimize the discriminator network in relation to it's own error.

        Args:
            optimizer_imp: The implementation of the optimizer to use.

        Returns:
            The operation to run to optimize the discriminator network.
        """
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/discriminator")

        #disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.disc_out, labels=tf.zeros_like(self.disc_out_sigmoid)), name="disc_fake_loss")
        #disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     logits=self.disc_gt, labels=tf.ones_like(self.disc_gt_sigmoid)), name="disc_real_loss")
        #disc_loss = tf.add(disc_fake, disc_real)

        disc_fake = tf.reduce_mean((self.disc_out - tf.zeros_like(self.disc_out_sigmoid))**2)
        disc_real = tf.reduce_mean((self.disc_gt - tf.ones_like(self.disc_gt_sigmoid))**2)
        disc_loss = tf.add(disc_fake, disc_real)

        # Add To Summary
        tf.summary.scalar("discriminator_loss_real", disc_real)
        tf.summary.scalar("discriminator_loss_fake", disc_fake)
        tf.summary.scalar("discriminator_loss", disc_loss)

        # Optimization
        #disc_opt = tf.train.AdamOptimizer(learning_rate=1e-8, beta1=0.9, beta2=0.999, epsilon=1e-8)
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
