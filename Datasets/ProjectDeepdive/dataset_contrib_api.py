import os

import tensorflow as tf

import Datasets.ProjectDeepdive.simulator as simulator

import dataset

class DatasetContribApi(dataset.Dataset):
    def __init__(self):
        parameters_list = ["train_path", "test_path", "input_size", "output_size", "turbidity_path",
                           "turbidity_size", "range_min", "range_max"]
        self.config_dict = self.open_config(parameters_list)
        self.batch_size = self.config_dict["batch_size"]
        self.input_size = self.config_dict["input_size"]
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
        self.range_min = self.config_dict["range_min"]
        self.range_max = self.config_dict["range_max"]
        self.sess = tf.Session()
        self.c, self.binf, self.range_array = simulator.acquireProperties(
            self.turbidity_path, self.turbidity_size, self.batch_size,
            self.range_min, self.range_max, self.sess)

    def _parse_function(self, record):
        features = {"image_raw": tf.FixedLenFeature((), tf.string),
                    "depth_raw": tf.FixedLenFeature((), tf.string)}
        parsed_features = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
        image.set_shape([self.input_size_prod])
        image = tf.cast(image, tf.float32) * (1. / 255)
        depth = tf.decode_raw(parsed_features['depth_raw'], tf.float32)
        depth.set_shape([self.output_size_prod])
        depth = tf.reshape(depth, self.output_size)
        image = tf.reshape(image, self.input_size)
        # depth = tf.cast(depth, tf.float32)
        return image, depth

    def next_batch_train(self, initial_step):
        """
        returns:
            a tuple(image, depths) where:
                image is a float tensor with shape [batch size] + input_size
                depth is a float tensor with shape [batch size] + depth_size
        """

        filenames = self.train_file

        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
        dataset = dataset.shuffle(buffer_size=3000)
        dataset = dataset.batch(self.config_dict["batch_size"])
        dataset = dataset.repeat(self.config_dict["num_epochs"])  # Repeat the input.
        dataset = dataset.skip(initial_step)

        iterator = dataset.make_one_shot_iterator()

        images, depths = iterator.get_next()
        #depths = tf.reshape(depths, [None] + self.output_size)
        #images = tf.reshape(images, [None] + self.input_size)
        images = simulator.applyTurbidity(images, depths, self.c, self.binf, self.range_array)
        tf.summary.image("depth", depths)
        tf.summary.image("image", images)
        return images, depths


    def next_batch_test(self):
        """
        returns:
            a tuple(image, depths) where:
                image is a float tensor with shape [batch size] + input_size
                depth is a float tensor with shape [batch size] + depth_size
        """

        filenames = self.test_file

        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
        dataset = dataset.batch(self.config_dict["batch_size"])
        iterator = dataset.make_initializable_iterator()
        initializer = iterator.initializer

        images, depths = iterator.get_next()
        # depths = tf.reshape(depths, [self.batch_size] + self.output_size)
        # images = tf.reshape(images, [self.batch_size] + self.input_size)
        images = simulator.applyTurbidity(images, depths, self.c, self.binf, self.range_array)
        tf.summary.image("depth", depths)
        tf.summary.image("image", images)
        return images, depths, initializer

    def get_num_samples(self, filename_list):
        num_samples = 0
        for filename in filename_list:
            num_samples += sum(1 for _ in tf.python_io.tf_record_iterator(filename))
        return num_samples

    def get_num_samples_test(self):
        return self.get_num_samples(self.test_file)

    def get_num_samples_train(self):
        return self.get_num_samples(self.train_file)
