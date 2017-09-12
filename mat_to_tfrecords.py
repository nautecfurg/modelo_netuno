import h5py
import numpy as np
import os
import tensorflow as tf
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def cropConvert(samples_pair, crop_dimension, original_dimension, writer):
    for image, depth in samples_pair:
        for i in range(18, original_dimension[0]-crop_dimension[0], crop_dimension[0]):
            for j in range(18, original_dimension[1]-crop_dimension[1], crop_dimension[1]):
                image_np = np.array(image[i:i + crop_dimension[0], j:j + crop_dimension[1], :])
                depth_np = np.array(depth[i:i + crop_dimension[0], j:j + crop_dimension[1]])
                image_raw = image_np.tostring()
                depth_raw = depth_np.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_raw),
                    'depth_raw': _bytes_feature(depth_raw)}))
                writer.write(example.SerializeToString())

    writer.close()

TF_RECORDS_TRAIN_FILENAME = 'Datasets/Data/nyu/tfrecords/train/nyu.tfrecords'
TF_RECORDS_VALIDATION_FILENAME = 'Datasets/Data/nyu/tfrecords/test/nyu.tfrecords'
MAT_FILENAME = "/home/joel/mestrado/modelo_netuno/Datasets/Data/nyu - labeled" + \
               "/nyu_depth_v2_labeled.mat"

TRAIN_PROPORTION = 0.9

mat_contents = h5py.File(MAT_FILENAME)

depths = mat_contents["depths"][:]
images = mat_contents["images"][:]
images = np.swapaxes(images,1,3)        # C x W x H => H x W x C
depths = np.swapaxes(depths,1,2)        # W x H => H x W

WRITER_TRAIN = tf.python_io.TFRecordWriter(TF_RECORDS_TRAIN_FILENAME)
WRITER_VALIDATION = tf.python_io.TFRecordWriter(TF_RECORDS_VALIDATION_FILENAME)

train_length = int(TRAIN_PROPORTION * images.shape[0])
validation_length = images.shape[0] - train_length

images_train = images[:train_length, :, :, :]
depths_train = depths[:train_length, :, :]

print(images_train.shape)
print(depths_train.shape)

images_validation = images[-validation_length:, :, :, :]
depths_validation = depths[-validation_length:, :, :]

dimensao_menor = (224, 224)
dimensao_maior = images.shape[1:3]
train_pair = zip(images_train, depths_train)
validation_pair = zip(images_validation, depths_validation)
cropConvert(train_pair, dimensao_menor, dimensao_maior, WRITER_TRAIN)
cropConvert(validation_pair, dimensao_menor,dimensao_maior,WRITER_VALIDATION)
