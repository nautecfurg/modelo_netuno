import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    std = tf.sqrt(tf.reduce_mean(tf.square(img)))
    return img/tf.maximum(std, eps)

def lap_normalize(img, n_levels=1):
    '''Perform the Laplacian pyramid normalization.'''
    k = np.float32([1,4,6,4,1])
    k = np.outer(k, k)
    k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)
    img = tf.expand_dims(img,0)

    levels = []
    for i in range(1, n_levels):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
        levels.append(hi)
        img=lo
    levels.append(img)
    tlevels=levels[::-1]
    tlevels = list(map(normalize_std, tlevels))

    img = tlevels[0]
    for hi in tlevels[1:]:
        img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img[0,:,:,:]

def tv_norm(x, beta):
    '''Computes the Total Variation Norm.'''
    p1=tf.square(x[:,1:,:-1,:]-x[:,:-1,:-1,:])
    p2=tf.square(x[:,:-1,1:,:]-x[:,:-1,:-1,:])
    norm=tf.reduce_sum(tf.pow((p1+p2),beta/2.0))
    return norm

def maximize_activation(input_size, x, ft_map, noise=True, step=1, iters=100, blur_every=4, blur_width=1, lap_levels=5, tv_lambda=0, tv_beta=2.0):
 '''Runs activation maximization'''

 t_score = tf.reduce_mean(ft_map)
 tv_x=tv_norm(x,tv_beta)
 t_grad = tf.gradients(t_score-tv_lambda*tv_x, x)[0]
 # laplacian pyramid gradient normalization
 grad_norm=lap_normalize(t_grad[0,:,:,:], lap_levels)

 # crates a gray image
 initial_img = np.zeros(input_size) + 0.5
 if noise:
  # adds noise to the initial image
  initial_img+=np.random.uniform(low=-0.5, high=0.5, size=input_size)

 images = np.empty((1,)+input_size)
 images[0] = initial_img.copy()

 sess = tf.get_default_session()
 for i in range(1,iters+1):
  feedDict={x: images}
  g, score = sess.run([grad_norm, t_score], feed_dict=feedDict)
  images[0] = images[0]+g*step
  # gaussian blur
  if blur_every:
   if i%blur_every==0:
    images[0] = gaussian_filter(images[0], sigma=blur_width)
# useless regularization methods
#  l2 decay
#  if config.decay:
#   images[0] = images[0]*(1-config.decay)
#  clip norm
#  if config.norm_pct_thrshld:
#   norms=np.linalg.norm(images[0], axis=2, keepdims=True)
#   n_thrshld=np.sort(norms, axis=None)[int(norms.size*config.norm_pct_thrshld)]
#   images[0]=images[0]*(norms>=n_thrshld)
#  #clip contribution
#  if config.contrib_pct_thrshld:
#   contribs=np.sum(images[0]*g[0], axis=2, keepdims=True)
#   c_thrshld=np.sort(contribs, axis=None)[int(contribs.size*config.contrib_pct_thrshld)]
#   images[0]=images[0]*(contribs>=c_thrshld)

 return images[0].astype(np.float32)
