import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import time
sys.path.append('..')
import Architectures.Layers.guidedfilter_grayscale as gg
import Architectures.Layers.guided_filter as gf
import Architectures.Layers.faster_guided_filter as fgf

def read_image(image_name):
    image = Image.open(image_name).convert("L")
    #image.show()
    image = np.asarray(image, dtype=np.float32)
    print(np.max(image))
    image = np.expand_dims(np.multiply(image, 1.0 / 255.0), -1)
    return image

def read_image_grayscale(image_name):
    image = Image.open(image_name).convert('L')
    #image.show()
    image = np.expand_dims(np.asarray(image, dtype=np.float32), -1)
    print(np.max(image))
    image = np.multiply(image, 1.0 / 255.0)
    return image

I_im = read_image('toy.bmp')
p_im = read_image_grayscale('toy-mask.bmp')
p_im = np.reshape(p_im, (1, p_im.shape[0], p_im.shape[1], 1))
I_im = np.reshape(I_im, (1, ) + I_im.shape)
pf = np.repeat(p_im, 16, 3)
If = np.repeat(I_im, 16, 3)
p = tf.placeholder("float", pf.shape, name="image")
I = tf.placeholder("float", If.shape, name="guide")


lr_shape = [int(224/4), int(219/4)]
print (lr_shape)

#q = guided_filter(I=I, p=p, feature_maps=16)
lr_I = tf.image.resize_images(If, lr_shape)
lr_p = tf.image.resize_images(pf, lr_shape)

feedDict = {p:pf, I:If}

print (lr_I, lr_p, I)
q = fgf.fast_guided_filter(lr_I, lr_p, I,
                           r=20 , eps=10**(-6),
                           nhwc=True)
#q = gf.guidedfilter(I, p, r=20 , eps=10**(-6))

sess = tf.InteractiveSession()
start_time = time.time()
q_np = sess.run(q,feed_dict=feedDict)
duration = time.time() - start_time
print (duration, "s")
print (np.min(q_np))
print (np.max(q_np))
q_np = np.clip(q_np, 0, 1)
print (q_np.shape)
q_np = np.reshape(q_np,q_np.shape[1:4])
im = Image.fromarray(np.uint8(q_np*255))

im.show()