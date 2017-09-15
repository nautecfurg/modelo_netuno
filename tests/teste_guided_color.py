import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import time
sys.path.append('..')
import Architectures.Layers.guidedfilter_color as gc


def read_image(image_name):
    image = Image.open(image_name)
    image.show()
    image = np.asarray(image, dtype=np.float32)
    print(np.max(image))
    image = np.multiply(image, 1.0 / 255.0)
    return image

def read_image_grayscale(image_name):
    image = Image.open(image_name).convert('L')
    image.show()
    image = np.asarray(image, dtype=np.float32)
    print(np.max(image))
    image = np.multiply(image, 1.0 / 255.0)
    return image

I_im = read_image('toy.bmp')
p_im = read_image_grayscale('toy-mask.bmp')
p_im = np.reshape(p_im, (1, p_im.shape[0], p_im.shape[1], 1))
I_im = np.reshape(I_im, (1, ) + I_im.shape)
p = tf.placeholder("float", p_im.shape, name="image")
I = tf.placeholder("float", I_im.shape, name="guide")

feedDict = {p:p_im, I:I_im}

q = gc.guidedfilter_color( I=I, p=p, r=60 , eps=10**(-6))
sess = tf.InteractiveSession()
start_time = time.time()
q_np = sess.run(q,feed_dict=feedDict)
duration = time.time() - start_time
print (duration)
print (np.min(q_np))
print (np.max(q_np))
q_np = np.clip(q_np, 0, 1)
q_np = np.reshape(q_np,q_np.shape[1:3])
im = Image.fromarray(np.uint8(q_np*255))

im.show()