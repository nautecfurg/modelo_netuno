import numpy as np
import tensorflow as tf

import Losses.discriminator as discriminator
import Losses.feature_loss as feature_loss
import Losses.mse as mse

import loss
import optimizer

class DeepLoss(loss.Loss):
    """This class is responsible for creating the DeepDive loss network, which is a
    mixture of FeatureLoss or MSE with the Discriminator Loss.

    The class allows the usage of the DeepDive loss.
    """

    def __init__(self):
        """This constructor initializes a new DeepLoss object.

        The function loads the parameters from the deep_loss json and creates a new 
        object based on them.

        Returns:
            Nothing.
        """
        parameters_list = []
        self.config_dict = self.open_config(parameters_list)

        # Initialize Losses
        self.mse_loss = mse.MSE()
        self.discriminator_loss = discriminator.DiscriminatorLoss()

    def evaluate(self, architecture_output, target_output):
        """This method evaluates the loss for the given image and it's ground-truth.

        The method models a discriminator neural network mixed with Feature Loss or MSE.

        Args:
            architecture_output: The image to input in the deep loss.

            target_output: The ground-truth image to input in the deep loss.

        Returns:
            The value of the deep loss.
        """
        return self.mse_loss.evaluate(architecture_output, target_output) + self.discriminator_loss.evaluate(architecture_output, target_output) / 2.0

    def train(self, optimizer_imp):
        """This method returns the training operation of the network.

        This method returns the training operation that is to be runned by tensorflow
        to minimize the deep dive network in relation to it's own error.

        Args:
            optimizer_imp: The implementation of the optimizer to use.

        Returns:
            The operation to run to optimize the deep dive network.
        """
        return self.discriminator_loss.train(optimizer_imp)

    def trainable(self):
        """This method tells whether this network is trainable or not.

        This method overrides the parent default method to make this network be trained on
        the main loop of the project.

        Returns:
            True, as the network is trainable.
        """
        return True
