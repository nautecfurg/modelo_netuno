import h5py
import numpy as np
import os
import tensorflow as tf
import dataset_manager as dm

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def crop_convert(samples_pair, crop_dimension, original_dimension, writer, initial_point):
    """This function crops the image pairs and store as tfrecord
    
    It takes the samples pair and from each pair extracts as much tiles as possible with size
    `crop_dimension` starting from the initial point. And while it crops, also stores the tiles 
    pairs in the specified writer. 

    Args:
        samples_pair: zipped object with input and output pair
        crop_dimension: list with the size of each tile that must be croped in the images
        on samples_pair
        original_dimension: list with the size of image in the samples_pair
        writer: tfrecord writer where the croped images will be stored
        initial_point: point were the tiles will start to be extracted from the original image

    """
    init_x, init_y = initial_point
    for image, depth in samples_pair:
        for i in range(init_x, original_dimension[0]-crop_dimension[0], crop_dimension[0]):
            for j in range(init_y, original_dimension[1]-crop_dimension[1], crop_dimension[1]):
                image_np = np.array(image[i:i + crop_dimension[0], j:j + crop_dimension[1], :])
                depth_np = np.array(depth[i:i + crop_dimension[0], j:j + crop_dimension[1]])
                image_np = image_np.tostring()
                depth_np = depth_np.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_np),
                    'depth_raw': _bytes_feature(depth_np)}))
                writer.write(example.SerializeToString())

    writer.close()

class NyuToTfrecords(dm.DatasetManager):
    """
        Dataset manager class designed to take nyu dataset and store as tfrecords.

        Besides it crops in a smaller size in order to fit the network.

    """

    def __init__(self):
        parameters_list = ["tf_records_train_filename", "tf_records_validation_filename",
                           "mat_filename", "train_proportion", "crop_dimension", "initial_point"]
        self.config_dict = self.open_config(parameters_list) # specifies needed configurations
                                                             # and open config

    def convert_data(self):
        mat_contents = h5py.File(self.config_dict["mat_filename"]) # load mat contents

        depths = mat_contents["depths"][:]       # take all depths
        images = mat_contents["images"][:]       # take all images
        images = np.swapaxes(images, 1, 3)        # C x W x H => H x W x C
        depths = np.swapaxes(depths, 1, 2)        # W x H => H x W
        train_length = int(self.config_dict["train_proportion"] * images.shape[0])
        validation_length = images.shape[0] - train_length

        images_train = images[:train_length, :, :, :]
        depths_train = depths[:train_length, :, :]

        images_validation = images[-validation_length:, :, :, :]
        depths_validation = depths[-validation_length:, :, :]

        train_pair = zip(images_train, depths_train)
        validation_pair = zip(images_validation, depths_validation)

        train_filename = self.config_dict["tf_records_train_filename"]
        validation_filename = self.config_dict["tf_records_validation_filename"]

        writer_train = tf.python_io.TFRecordWriter(train_filename)
        writer_validation = tf.python_io.TFRecordWriter(validation_filename)

        crop_dimension = self.config_dict["crop_dimension"]
        original_dimension = images.shape[1:3]
        init_p = self.config_dict["initial_point"]

        crop_convert(train_pair, crop_dimension, original_dimension, writer_train, init_p)
        crop_convert(validation_pair, crop_dimension, original_dimension, writer_validation,
                     init_p)
