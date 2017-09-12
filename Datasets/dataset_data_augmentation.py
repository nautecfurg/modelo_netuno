from __future__ import absolute_import, division

import glob
import gzip
import os
import random
from time import time

import numpy as np

import dataset
import leveldb
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib


def readImageFromDB(db, key, size):
    image = np.reshape(np.fromstring(db.Get(key), dtype=np.float32), size)
    return image

class DataSet(object):
    def __init__(self, images_key, input_size, depth_size, num_examples,
                 db, validation, invert, rotate):
        self._db = db
        self._is_validation = validation
        self._num_examples = num_examples
        self._images_key = images_key
        random.shuffle(self._images_key)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._input_size = input_size
        self._depth_size = depth_size
        self.invert = invert
        self.rotate = rotate

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if batch_size > (self._num_examples - self._index_in_epoch):
            # Finished epoch
            print('end epoch')
            self._epochs_completed += 1
            # Shuffle the data
            """ Shufling all the Images with a single permutation """
            random.shuffle(self._images_key)
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        images = np.empty((batch_size, self._input_size[0],
                           self._input_size[1], self._input_size[2]))
        if len(self._depth_size) == 2:
            self._depth_size = (self._depth_size[0], self._depth_size[1], 1)
        depths = np.empty((batch_size, self._depth_size[0],
                           self._depth_size[1], self._depth_size[2]))
        for n in range(batch_size):
            key = self._images_key[start+n]
            if self.rotate:
                rotation = key & 3
                key = int(key/4)
            else:
                rotation = 0
            if self.invert:
                inversion = key & 1
                key = int(key/2)
            else:
                inversion = 0
            if self._is_validation:
                images[n] = readImageFromDB(self._db, 'val'+str(key), self._input_size)
                depths[n] = readImageFromDB(self._db, 'val'+str(key)+"depth", self._depth_size)
            else:
                images[n] = readImageFromDB(self._db, str(key), self._input_size)
                depths[n] = readImageFromDB(self._db, str(key)+"depth", self._depth_size)

            images[n] = np.rot90(images[n], rotation)
            depths[n] = np.rot90(depths[n], rotation)

            if inversion:
                images[n] = np.fliplr(images[n])
                depths[n] = np.fliplr(depths[n])
        return images, depths


class DatasetDataAugmentation(dataset.Dataset):
    def __init__(self):
        self.input_size = self.config_dict['input_size']
        self.depth_size = self.config_dict['depth_size']
        self.db = leveldb.LevelDB(self.config_dict['leveldb_path'] + 'db')
        self.num_examples = int(self.db.Get('num_examples'))
        self.num_examples_val = int(self.db.Get('num_examples_val'))
        if self.config_dict["invert"]:
            self.num_examples = self.num_examples * 2
            self.num_examples_val = self.num_examples_val * 2
        if self.config_dict["rotate"]:
            self.num_examples = self.num_examples * 4
            self.num_examples_val = self.num_examples_val * 4
        self.images_key = range(self.num_examples)
        self.images_key_val = range(self.num_examples_val)

        self.train = DataSet(self.images_key, self.config_dict["input_size"],
                             self.config_dict["depth_size"], self.num_examples,
                             self.db, validation=False, invert=self.config_dict["invert"],
                             rotate=self.config_dict["rotate"])
        self.validation = DataSet(self.images_key_val, self.config_dict["input_size"],
                                  self.config_dict["depth_size"], self.num_examples_val,
                                  self.db, validation=True,
                                  invert=self.config_dict["invert"], rotate=self.config_dict["rotate"])
    def next_batch_train(self, batch_size, num_epochs=None):
        images, depths = self.train.next_batch(batch_size, num_epochs)
        return images, depths

    def next_batch_validation(self, batch_size, num_epochs):
        images, depths = self.validation.next_batch(batch_size, num_epochs)
        return images, depths

    def get_n_images_train(self):
        return self.num_examples

    def get_n_images_validation(self):
        return self.num_examples_val
