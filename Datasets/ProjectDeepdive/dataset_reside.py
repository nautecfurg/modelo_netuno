import os

import tensorflow as tf

import Datasets.ProjectDeepdive.simulator as simulator

import dataset

class DatasetReside(dataset.Dataset):
    def __init__(self):
        parameters_list = ["train_path", "test_path", "input_size", "clear_size", "output_size"]
        self.config_dict = self.open_config(parameters_list)
        self.batch_size = self.config_dict["batch_size"]
        self.input_size = self.config_dict["input_size"]
        self.clear_size = self.config_dict["clear_size"]
        self.input_size_prod = self.input_size[0] * self.input_size[1] * self.input_size[2]
        self.clear_size_prod = self.clear_size[0] * self.clear_size[1] * self.clear_size[2]
        self.train_path = self.config_dict["train_path"]
        self.train_file = [self.train_path + f for f in os.listdir(self.train_path)]
        self.test_path = self.config_dict["test_path"]
        self.test_file = [self.test_path + f for f in os.listdir(self.test_path)]

    def _parse_function(self, record):
        features = {"image_raw": tf.FixedLenFeature((), tf.string),
                    "clear_raw": tf.FixedLenFeature((), tf.string)}
        parsed_features = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
        image.set_shape([self.input_size_prod])
        image = tf.cast(image, tf.float32) * (1. / 255)
        clear = tf.decode_raw(parsed_features['clear_raw'], tf.uint8)
        image = tf.cast(clear, tf.float32) * (1. / 255)
        clear.set_shape([self.clear_size_prod])
        clear = tf.reshape(clear, self.clear_size)
        image = tf.reshape(image, self.input_size)
        return image, clear

    def next_batch_train(self, initial_step):
        """
        returns:
            a tuple(image, clear) where:
                image is a float tensor with shape [batch size] + input_size
                clear is a float tensor with shape [batch size] + clear_size
        """

        filenames = self.train_file

        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
        dataset = dataset.shuffle(buffer_size=3000)
        dataset = dataset.batch(self.config_dict["batch_size"])
        dataset = dataset.repeat(self.config_dict["num_epochs"])  # Repeat the input.
        dataset = dataset.skip(initial_step)

        iterator = dataset.make_one_shot_iterator()

        image, clear = iterator.get_next()
        tf.summary.image("clear", clear)
        tf.summary.image("image", image)
        return image, clear


    def next_batch_test(self):
        """
        returns:
            a tuple(image, clear) where:
                image is a float tensor with shape [batch size] + input_size
                clear is a float tensor with shape [batch size] + clear_size
        """

        filenames = self.test_file

        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
        dataset = dataset.batch(self.config_dict["batch_size"])

        iterator = dataset.make_initializable_iterator()
        initializer = iterator.initializer

        image, clear = iterator.get_next()
        tf.summary.image("image", image)
        tf.summary.image("clear", clear)
        return image, clear, initializer

    def get_num_samples(self, filename_list):
        num_samples = 0
        for filename in filename_list:
            num_samples += sum(1 for _ in tf.python_io.tf_record_iterator(filename))
        return num_samples

    def get_num_samples_test(self):
        return self.get_num_samples(self.test_file)

    def get_num_samples_train(self):
        return self.get_num_samples(self.train_file)
