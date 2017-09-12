#pylint: disable=E0611,E0401,C0325
import os
import random

import tensorflow as tf

import Datasets.TransmissionNet.simulator as simulator

import dataset

class DatasetTransmission(dataset.Dataset):
    def __init__(self):
        parameters_list = ["train_path", "test_path", "input_size", "output_size", "turbidity_path",
                           "turbidity_size", "patch_size", "trans_minval", "trans_maxval"]
        self.config_dict = self.open_config(parameters_list)
        self.batch_size = self.config_dict["batch_size"]
        self.input_size = self.config_dict["input_size"]
        self.patch_size = self.config_dict["patch_size"]
        self.input_size_prod = self.input_size[0] * self.input_size[1] * self.input_size[2]
        self.output_size = self.config_dict["output_size"]
        self.output_size_prod = self.output_size[0] * self.output_size[1] * self.output_size[2]
        self.train_path = self.config_dict["train_path"]
        self.train_file = [self.train_path + f for f in os.listdir(self.train_path)]
        self.test_path = self.config_dict["test_path"]
        self.test_file = [self.test_path + f for f in os.listdir(self.test_path)]
        #Simulator attributes
        self.turbidity_path = self.config_dict["turbidity_path"]
        self.turbidity_size = tuple(self.config_dict["turbidity_size"])
        self.sess = tf.Session()
        _, self.binf, _ = simulator.acquireProperties(
            self.turbidity_path, self.turbidity_size, self.batch_size, 0, 0, self.sess)

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
            })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.input_size_prod])
        image = tf.cast(image, tf.float32) * (1. / 255)

        return image

    def _parse_function(self, record):
        features = {"image_raw": tf.FixedLenFeature((), tf.string),}
                    
        parsed_features = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
        image.set_shape([self.input_size_prod])
        image = tf.cast(image, tf.float32) * (1. / 255)
        image = tf.reshape(image, self.input_size)
        # depth = tf.cast(depth, tf.float32)
        return image

    def random_transmissions(self, batch_size):
        minval = self.config_dict["trans_minval"]
        maxval = self.config_dict["trans_maxval"]
        transmissions = tf.random_uniform([batch_size], minval=minval, maxval=maxval)
        return transmissions

    def next_batch_train(self, initial_step):
        """
        args:
            batch_size:
                number of examples per returned batch
            num_epochs:
                number of time to read the input data

        returns:
            a tuple(image, transmissions) where:
                image is a float tensor with shape [batch size] + patch_size
                transmissions is a float tensor with shape [batch size]
        """

        filenames = self.train_file

        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
        dataset = dataset.shuffle(buffer_size=3000)
        dataset = dataset.batch(self.config_dict["batch_size"])
        dataset = dataset.repeat(self.config_dict["num_epochs"])  # Repeat the input.
        dataset = dataset.skip(initial_step)

        iterator = dataset.make_one_shot_iterator()

        images = iterator.get_next()

        size_x = self.config_dict['patch_size'][0]
        size_y = self.config_dict['patch_size'][1]
        offset_x = random.randint(0, self.input_size[0] - size_x - 1)
        offset_y = random.randint(0, self.input_size[0] - size_y - 1)

        images = images[:, offset_x:offset_x + size_x, offset_y:offset_y+size_y]
        transmissions = self.random_transmissions(self.batch_size)
        images = simulator.applyTurbidityTransmission(images, self.binf, transmissions)
        tf.summary.image("image", images)
        return images, transmissions

    #TODO: resolver o problema do batch size variavel
    def next_batch_test(self):
        """
        returns:
            a tuple(image, transmissions) where:
                image is a float tensor with shape [batch size] + input_size
                transmissions is a float tensor with shape [batch size]
        """

        filenames = self.test_file

        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
        dataset = dataset.batch(self.config_dict["batch_size"])
        iterator = dataset.make_initializable_iterator()
        initializer = iterator.initializer

        images = iterator.get_next()

        size_x = self.config_dict['patch_size'][0]
        size_y = self.config_dict['patch_size'][1]
        offset_x = random.randint(0, self.input_size[0] - size_x - 1)
        offset_y = random.randint(0, self.input_size[0] - size_y - 1)

        images = images[:, offset_x:offset_x + size_x, offset_y:offset_y+size_y]
        # depths = tf.reshape(depths, [self.batch_size] + self.output_size)
        # images = tf.reshape(images, [self.batch_size] + self.input_size)

        transmissions = self.random_transmissions(self.batch_size)
        images = simulator.applyTurbidityTransmission(images, self.binf, transmissions)
        tf.summary.image("image", images)
        return images, transmissions, initializer
