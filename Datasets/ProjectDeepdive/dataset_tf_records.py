import os

import tensorflow as tf

import Datasets.ProjectDeepdive.simulator as simulator

import dataset

class DatasetTfRecords(dataset.Dataset):
    def __init__(self):
        parameters_list = ["tfr_path", "input_size", "output_size", "turbidity_path",
                           "turbidity_size", "range_min", "range_max"]
        self.config_dict = self.open_config(parameters_list)
        self.batch_size = self.config_dict["batch_size"]
        self.input_size = self.config_dict["input_size"]
        self.input_size_prod = self.input_size[0] * self.input_size[1] * self.input_size[2]
        self.output_size = self.config_dict["output_size"]
        self.output_size_prod = self.output_size[0] * self.output_size[1] * self.output_size[2]
        self.tfr_path = self.config_dict["tfr_path"]
        self.train_file = [self.tfr_path + f for f in os.listdir(self.tfr_path)]
        self.num_file = len(self.train_file)
        self.validation_file = [self.tfr_path + f for f in os.listdir(self.tfr_path)]
        self.num_files_val = len(self.validation_file)
        #Simulator attributes
        self.turbidity_path = self.config_dict["turbidity_path"]
        self.turbidity_size = tuple(self.config_dict["turbidity_size"])
        self.range_min = self.config_dict["range_min"]
        self.range_max = self.config_dict["range_max"]
        self.sess = tf.Session()
        self.c, self.binf, self.range_array = simulator.acquireProperties(
            self.turbidity_path, self.turbidity_size, self.batch_size, self.range_min, self.range_max, self.sess)

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'depth_raw': tf.FixedLenFeature([], tf.string),
            })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.input_size_prod])
        image = tf.cast(image, tf.float32) * (1. / 255)
        depth = tf.decode_raw(features['depth_raw'], tf.uint8)
        depth.set_shape([self.output_size_prod])
        depth = tf.cast(depth, tf.float32) * (1. / 255)

        return image, depth

    def next_batch_train(self):
        """
        args:
            train:
                true: training
                false: validation
            batch_size:
                number of examples per returned batch
            num_epochs:
                number of time to read the input data

        returns:
            a tuple(image, depths) where:
                image is a float tensor with shape [batch size] + input_size
                depth is a float tensor with shape [batch size] + depth_size
        """

        filename = self.train_file
        print(filename)

        filename_queue = tf.train.string_input_producer(
            filename, num_epochs=self.config_dict["num_epochs"])

        image, depth = self.read_and_decode(filename_queue)

        images, depths = tf.train.shuffle_batch(
            [image, depth], batch_size=self.config_dict["batch_size"],
            num_threads=self.config_dict["num_threads"],
            capacity=100+ 3 * self.config_dict["batch_size"],
            min_after_dequeue=100
            )
        depths = tf.reshape(depths, [self.batch_size] + self.output_size)
        images = tf.reshape(images, [self.batch_size] + self.input_size)
        images = simulator.applyTurbidity(images, depths, self.c, self.binf, self.range_array)
        tf.summary.image("depth", depths)
        tf.summary.image("image", images)
        return images, depths
