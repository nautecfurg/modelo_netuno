import tensorflow as tf

def MF(A,n):
  result=tf.pow(tf.truediv(tf.shape(A)[-1],n),0.5)
  return tf.cast(tf.expand_dims(result,0),tf.float32)

def fast_choquet_pooling(images, ksizes, strides, rates, padding, q=0.5):
  #funcao mais rapida, mas sem possibilidade de mudar a metrica fuzzy
  patch_depth=ksizes[0]*ksizes[1]*ksizes[2]*ksizes[3]
  patches=tf.extract_image_patches(images=images, ksizes=ksizes, strides=strides, rates=rates, padding=padding)  
  sh=tf.shape(patches)
  n_channels=images.shape[3]
  patches=tf.reshape(patches,[sh[0],sh[1],sh[2],patch_depth,n_channels])
  patches=tf.transpose(patches, [0,1,2,4,3])
  p_sorted,_=tf.nn.top_k(patches, patch_depth, sorted=True)
  #aumenta o peso do maior valor da janela em 20%
  #p_sorted=p_sorted*tf.pad(tf.constant([1.2]), [[0,patch_depth-1]], constant_values=1.0)
  p_sorted=p_sorted[:,:,:,:,::-1]
  p_diff=p_sorted-tf.pad(p_sorted, [[0,0],[0,0],[0,0],[0,0],[1,0]])[:,:,:,:,:-1]
  f=tf.range(start=(tf.shape(p_sorted)[4]), limit=0, delta=-1)
  f=tf.cast(tf.pow(tf.truediv(f,patch_depth),q),tf.float32)
  S=p_diff*f 
  return tf.reduce_sum(S, axis=4)

def fast_choquet_pooling_trainable(images, ksizes, strides, rates, padding):
  #funcao mais rapida, mas sem possibilidade de mudar a metrica fuzzy
  q=tf.Variable(initial_value=tf.random_uniform(shape=(), dtype=tf.float64), dtype=tf.float64, trainable=True)
  patch_depth=ksizes[0]*ksizes[1]*ksizes[2]*ksizes[3]
  patches=tf.extract_image_patches(images=images, ksizes=ksizes, strides=strides, rates=rates, padding=padding)  
  sh=tf.shape(patches)
  n_channels=images.shape[3]
  patches=tf.reshape(patches,[sh[0],sh[1],sh[2],patch_depth,n_channels])
  patches=tf.transpose(patches, [0,1,2,4,3])
  p_sorted,_=tf.nn.top_k(patches, patch_depth, sorted=True)
  #aumenta o peso do maior valor da janela em 20%
  #p_sorted=p_sorted*tf.pad(tf.constant([1.2]), [[0,patch_depth-1]], constant_values=1.0)
  p_sorted=p_sorted[:,:,:,:,::-1]
  p_diff=p_sorted-tf.pad(p_sorted, [[0,0],[0,0],[0,0],[0,0],[1,0]])[:,:,:,:,:-1]
  f=tf.range(start=(tf.shape(p_sorted)[4]), limit=0, delta=-1)
  f=tf.cast(tf.pow(tf.truediv(f,patch_depth),q),tf.float32)
  S=p_diff*f 
  return tf.reduce_sum(S, axis=4)

def choquet_pooling(images, ksizes, strides, rates, padding, fuzzy):
  patch_depth=ksizes[0]*ksizes[1]*ksizes[2]*ksizes[3]
  patches=tf.extract_image_patches(images=images, ksizes=ksizes, strides=strides, rates=rates, padding=padding)  
  sh=tf.shape(patches)
  n_channels=images.shape[3]
  patches=tf.reshape(patches,[sh[0],sh[1],sh[2],patch_depth,n_channels])
  patches=tf.transpose(patches, [0,1,2,4,3])
  p_sorted,p_index=tf.nn.top_k(patches, patch_depth, sorted=True)
  #aumenta o peso do maior valor da janela em 20%
  #p_sorted=p_sorted*tf.pad(tf.constant([1.2]), [[0,patch_depth-1]], constant_values=1.0)
  p_sorted=p_sorted[:,:,:,:,::-1]
  p_index=tf.expand_dims(p_index[:,:,:,:,::-1],-2)
  #T=tf.range(0,tf.shape(p_sorted)[4])
  #coloca T no mesmo shape de p_index, necessario para a funcao tf.sets.set_difference
  #T=tf.tile(T, [sh[0]*sh[1]*sh[2]*n_channels])
  #T=tf.reshape(T, [sh[0],sh[1],sh[2],n_channels,-1])
  p_diff=p_sorted-tf.pad(p_sorted, [[0,0],[0,0],[0,0],[0,0],[1,0]])[:,:,:,:,:-1]
  #S=tf.zeros(tf.shape(p_sorted)[:-1])
  #for i in range(patch_depth):
    #S=S+p_diff[:,:,:,:,i]*fuzzy(p_index[:,:,:,:,:,i:],patch_depth)
    #T=tf.sets.set_difference(T,p_index[:,:,:,:,:,i])
  #return S
  f=fuzzy(p_index,patch_depth)
  for i in range(1,patch_depth):
    f=tf.concat([f,fuzzy(p_index[:,:,:,:,:,i:],patch_depth)],-1)
  S=p_diff*f 
  return tf.reduce_sum(S, axis=4)

