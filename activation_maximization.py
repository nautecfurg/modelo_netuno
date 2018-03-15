import tensorflow as tf
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate as im_rotate
from scipy.ndimage import zoom

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

def clipped_zoom(img, zoom_factor, **kwargs):
#code from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def maximize_activation(input_size, x, ft_map, noise=True, step=1, iters=100, blur_every=4, blur_width=1, lap_levels=5, 
                                               tv_lambda=0, tv_beta=2.0, jitter=False, scale=False, rotate=False):
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

  # padding the input by 16 pixels to avoid edge artefacts
  tmp_img = np.pad(images[0], ((16, 16),(16, 16),(0,0)), 'wrap')

  # jitter (translation) regularization
  if jitter:
    jitter_px=np.random.randint(low=1, high=16)
    jitter_py=np.random.randint(low=1, high=16)
    tmp_img=np.roll(tmp_img, (jitter_px,jitter_py), axis=(0,1))

  # scaling regularization
  if scale:
    scaling_factors=[1, 0.975, 1.025, 0.95, 1.05]
    scale=scaling_factors[np.random.randint(low=0, high=4)]
    tmp_img=clipped_zoom(tmp_img, scale)

  # rotation regularization
  if rotate:
    rotate_angle=np.random.randint(low=-5, high=5)
    tmp_img=im_rotate(tmp_img, rotate_angle, reshape=False)

  # jittering a second time
  if jitter:
    jitter_px=np.random.randint(low=1, high=8)
    jitter_py=np.random.randint(low=1, high=8)
    tmp_img=np.roll(tmp_img, (jitter_px,jitter_py), axis=(0,1))

  # remove padding
  images[0] = tmp_img[16:-16,16:-16,:]
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
