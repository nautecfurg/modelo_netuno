import tensorflow as tf
import numpy as np
import Image
import sys
sys.path.append('..')
import Architectures.Layers.guidedfilter_grayscale as gf
from guidedfilter_grayscale import *


def read_image(image_name):
    image = Image.open(image_name)
    image.show()
    image = np.asarray(image, dtype=np.float32)
    print(np.max(image))
    image = np.multiply(image, 1.0 / 255.0)
    return image

im = read_image('cat.png')
p = tf.placeholder("float", (1, im.shape[0], im.shape[1], 1), name="image")
I = tf.placeholder("float", (1, im.shape[0], im.shape[1], 1), name="guide")
im = np.reshape(im, (1, im.shape[0], im.shape[1], 1))
feedDict = {p:im, I:im}

q = gf.guidedfilter( I=I, p=p, r=4 , eps=0.16)
sess = tf.InteractiveSession()
q_np = sess.run(q,feed_dict=feedDict)
print(np.max(q_np))
q_np = np.reshape(q_np,q_np.shape[1:3])
im = Image.fromarray(np.uint8(q_np*255))

im.show()