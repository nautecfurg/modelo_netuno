from PIL import Image
import os
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

haze_levels = 35

TF_RECORDS_TRAIN_FILENAME = 'Datasets/Data/reside/tfrecords/train/reside.tfrecords'
TF_RECORDS_VALIDATION_FILENAME = 'Datasets/Data/reside/tfrecords/test/reside.tfrecords'

WRITER_TRAIN = tf.python_io.TFRecordWriter(TF_RECORDS_TRAIN_FILENAME)
WRITER_VALIDATION = tf.python_io.TFRecordWriter(TF_RECORDS_VALIDATION_FILENAME)

IMAGE_PATH = '/home/nautec/Downloads/reside/OTS/OTS/'
CLEAR_PATH = '/home/nautec/Downloads/reside/clear_images/'

IMAGES = sorted([IMAGE_PATH + f for f in os.listdir(IMAGE_PATH)])
CLEARS = sorted([CLEAR_PATH + f for f in os.listdir(CLEAR_PATH)])

train_length = int(0.9 * len(IMAGES))
validation_length = len(IMAGES) - train_length
images_train = IMAGES[:train_length*haze_levels]
clears_train = CLEARS[:train_length]

images_validation = IMAGES[-(validation_length*haze_levels):]
clears_validation = CLEARS[-validation_length:]

#FILENAME_TRAIN_PAIRS = zip(images_train, depths_train)
#FILENAME_VALIDATION_PAIRS = zip(images_validation, depths_validation)

count = 0
i = 0
for clr_path in clears_train:
    clear = np.array(Image.open(clr_path))
    clear_raw = clear.tostring()
    for j in xrange(i, i + haze_levels):
        img_path = images_train[j]
        if i == j:
            print img_path.split("/")[-1], clr_path.split("/")[-1]
        img = np.array(Image.open(img_path))
        image_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'clear_raw': _bytes_feature(clear_raw)}))
        WRITER_TRAIN.write(example.SerializeToString())
    i += haze_levels
        
WRITER_TRAIN.close()

i = 0
for clr_path in clears_validation:
    clear = np.array(Image.open(clr_path))
    clear_raw = clear.tostring()
    for j in xrange(i, i + haze_levels):
        img_path = images_validation[j]
        if i == j:
            print img_path.split("/")[-1], clr_path.split("/")[-1]
        img = np.array(Image.open(img_path))
        image_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'clear_raw': _bytes_feature(clear_raw)}))
        WRITER_VALIDATION.write(example.SerializeToString())
    i += haze_levels

WRITER_VALIDATION.close()
