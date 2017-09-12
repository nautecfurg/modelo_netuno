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


TF_RECORDS_TRAIN_FILENAME = 'Datasets/Data/nyu/tfrecords/train/nyu.tfrecords'
TF_RECORDS_VALIDATION_FILENAME = 'Datasets/Data/nyu/tfrecords/test/nyu.tfrecords'

WRITER_TRAIN = tf.python_io.TFRecordWriter(TF_RECORDS_TRAIN_FILENAME)
WRITER_VALIDATION = tf.python_io.TFRecordWriter(TF_RECORDS_VALIDATION_FILENAME)

IMAGE_PATH = 'Datasets/Data/nyu/images/'
#DEPTH_PATH = 'Datasets/Data/nyu/depths/'

IMAGES = sorted([IMAGE_PATH + f for f in os.listdir(IMAGE_PATH)])
#DEPTHS = sorted([DEPTH_PATH + f for f in os.listdir(DEPTH_PATH)])

train_length = int(0.9 * len(IMAGES))
validation_length = len(IMAGES) - train_length
images_train = IMAGES[:train_length]
#depths_train = DEPTHS[:train_length]

images_validation = IMAGES[-validation_length:]
#depths_validation = DEPTHS[-validation_length:]

#FILENAME_TRAIN_PAIRS = zip(images_train, depths_train)
#FILENAME_VALIDATION_PAIRS = zip(images_validation, depths_validation)

for img_path in images_train:#, depth_path in FILENAME_TRAIN_PAIRS:
    img = np.array(Image.open(img_path))
    #depth = np.array(Image.open(depth_path)) * (255.0/65535.0)
    #depth = depth.astype(np.uint8)
    image_raw = img.tostring()
    #depth_raw = depth.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        }))#    'depth_raw': _bytes_feature(depth_raw)}))
    WRITER_TRAIN.write(example.SerializeToString())

WRITER_TRAIN.close()


for img_path in images_validation:#, depth_path in FILENAME_VALIDATION_PAIRS:
    img = np.array(Image.open(img_path))
    #depth = np.array(Image.open(depth_path)) * (255.0/65535.0)
    #depth = depth.astype(np.uint8)
    image_raw = img.tostring()
    #depth_raw = depth.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        }))#'depth_raw': _bytes_feature(depth_raw)}))
    WRITER_VALIDATION.write(example.SerializeToString())

WRITER_VALIDATION.close()
