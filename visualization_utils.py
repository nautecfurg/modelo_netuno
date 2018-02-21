import tensorflow as tf
import numpy as np
import math
from PIL import Image
import os
#from features_optimization import normalize_std


def save_optimized_image_to_disk(opt_output, channel,n_channels,key,path):
    opt_output_rescaled = (opt_output - opt_output.min())
    opt_output_rescaled *= (255/(opt_output_rescaled.max()+0.0001))
    im = Image.fromarray(opt_output_rescaled.astype(np.uint8))
    file_name="opt_"+str(channel).zfill(len(str(n_channels)))+".bmp"
    folder_name=path+"/feature_maps/"+key+"/optimization"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    im.save(folder_name+"/"+file_name)

def save_images_to_disk(result_imgs,input_imgs,gt_imgs, path):
    result_imgs=(np.clip(result_imgs, 0, 1) * 255).round().astype(np.uint8)
    for j in xrange(result_imgs.shape[0]):
        im = Image.fromarray(result_imgs[j])
        file_name="output.bmp"
        im_folder=str(j).zfill(len(str(result_imgs.shape[0])))
        folder_name=path+"/output/"+im_folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        im.save(folder_name+"/"+file_name)

    input_imgs=(np.clip(input_imgs, 0, 1) * 255).round().astype(np.uint8)
    for j in xrange(input_imgs.shape[0]):
        im = Image.fromarray(input_imgs[j])
        file_name="input.bmp"
        im_folder=str(j).zfill(len(str(input_imgs.shape[0])))
        folder_name=path+"/input/"+im_folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        im.save(folder_name+"/"+file_name)

    gt_imgs=(np.clip(gt_imgs, 0, 1) * 255).round().astype(np.uint8)
    for j in xrange(gt_imgs.shape[0]):
        im = Image.fromarray(gt_imgs[j])
        file_name="ground_truth.bmp"
        im_folder=str(j).zfill(len(str(gt_imgs.shape[0])))
        folder_name=path+"/ground_truth/"+im_folder
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        im.save(folder_name+"/"+file_name) 

def save_feature_maps_to_disk(feature_maps, weights, deconv, feature_names,path):
    for ft, w, d, key in zip(feature_maps, weights, deconv, feature_names):
        for l in xrange(ft.shape[3]):
            ft_img=ft[:,:,:,l]
            ft_img = (ft_img - ft_img.min())
            ft_img*=(255/(ft_img.max()+0.0001))
            for k in xrange(ft.shape[0]):		
                ch_img=ft_img[k,:,:].astype(np.uint8) 
                im = Image.fromarray(ch_img)
                file_name=str(l).zfill(len(str(ft.shape[3])))+".bmp"
                im_folder=str(k).zfill(len(str(ft.shape[0])))
                folder_name=path+"/feature_maps/"+key+"/"+im_folder
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                im.save(folder_name+"/"+file_name)
        if w is not None:
            kernel=w.eval()
            kernel_img = (kernel - kernel.min())
            kernel_img*=(255/(kernel_img.max()+0.0001))
            for k in xrange(kernel_img.shape[3]):
                im = Image.fromarray(kernel_img[:,:,:,k].astype(np.uint8))
                k_file_name="W_"+str(k).zfill(len(str(ft.shape[3])))+".bmp"
                k_folder_name=path+"/feature_maps/"+key+"/kernels"
                if not os.path.exists(k_folder_name):
                    os.makedirs(k_folder_name)
                im.save(k_folder_name+"/"+k_file_name)
        if d is not None:
            for i in xrange(d.shape[4]):
                for j in xrange(d.shape[0]):
                    d_img=d[j,:,:,:,i]
                    d_img = (d_img-d_img.min())
                    d_img*=(255/(d_img.max()+0.0001))
                    deconv_img=d_img.astype(np.uint8) 
                    im = Image.fromarray(deconv_img)
                    file_name=str(i).zfill(len(str(d.shape[4])))+".bmp"
                    im_folder=str(j).zfill(len(str(d.shape[0])))
                    folder_name=path+"/deconv/"+key+"/"+im_folder
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    im.save(folder_name+"/"+file_name)


def save_max_activations_to_disk(max_activation, feature_names,path):
    for actv, key in zip(max_activation, feature_names):
        for ch in xrange(actv[0].shape[4]):
            for n in xrange(actv[0].shape[0]):
                actv_img=actv[1][n,:,:,ch]
                actv_img = (actv_img-actv_img.min())
                actv_img*=(255/(actv_img.max()+0.0001))
                max_actv_img=actv_img.astype(np.uint8)
                input_img=actv[0][n,:,:,:,ch]
                input_img = (input_img-input_img.min())
                input_img*=(255/(input_img.max()+0.0001))
                max_actv_input_img=input_img.astype(np.uint8) 
                actv_im = Image.fromarray(max_actv_img)
                actv_file_name=str(n).zfill(len(str(actv[0].shape[0]-1)))+".bmp"
                input_im = Image.fromarray(max_actv_input_img)
                input_file_name="input_"+str(n).zfill(len(str(actv[0].shape[0]-1)))+".bmp"
                folder_name=path+"/feature_maps/"+key+"/max_activations/"+str(ch).zfill(len(str(actv[0].shape[4]-1)))
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                actv_im.save(folder_name+"/"+actv_file_name)
                input_im.save(folder_name+"/"+input_file_name)

def put_features_on_grid(features):
    iy=tf.shape(features, out_type=tf.int32)[1]
    ix=tf.shape(features, out_type=tf.int32)[2]
    n_ch=tf.cast(tf.shape(features, out_type=tf.int32)[3], tf.float32)
    b_size=tf.shape(features, out_type=tf.int32)[0]
    square_size=tf.cast(tf.ceil(tf.sqrt(n_ch)),tf.float32)
    z_pad=tf.cast(square_size**2-n_ch, tf.int32)
    black=tf.minimum(0.0,tf.reduce_min(features))
    pad=1+tf.cast((ix/64), tf.int32)
    square_size=tf.cast(square_size, tf.int32)
    features = tf.pad(features, [[0,0],[0,0],[0,0],[0,z_pad]], mode='constant',constant_values=black)
    features = tf.reshape(features,[b_size,iy,ix,square_size,square_size])
    features = tf.pad(features, [[0,0],[pad,0],[pad,0],[0,0],[0,0]], mode='constant',constant_values=black)
    iy+=pad
    ix+=pad
    features = tf.transpose(features,(0,3,1,4,2))
    features = tf.reshape(features,[-1,square_size*iy,square_size*ix,1])
    return tf.pad(features, [[0,0],[0,pad],[0,pad],[0,0]], mode='constant',constant_values=black)

def put_kernels_on_grid(kernels):
    iy=tf.shape(kernels, out_type=tf.int32)[0]
    ix=tf.shape(kernels, out_type=tf.int32)[1]
    n_ch=tf.cast(tf.shape(kernels, out_type=tf.int32)[3], tf.float32)
    square_size=tf.cast(tf.ceil(tf.sqrt(n_ch)),tf.float32)
    z_pad=tf.cast(square_size**2-n_ch, tf.int32)
    black=tf.minimum(0.0,tf.reduce_min(kernels))
    pad=1+tf.cast((ix/64), tf.int32)
    kernels = tf.pad(kernels, [[0,0],[0,0],[0,0],[0,z_pad]], mode='constant',constant_values=black)
    kernels = tf.transpose(kernels,(0,1,3,2))
    kernels = tf.reshape(kernels,[iy,ix,square_size,square_size,3])
    kernels = tf.pad(kernels, [[pad,0],[pad,0],[0,0],[0,0],[0,0]], mode='constant',constant_values=black)
    iy+=pad
    ix+=pad
    kernels = tf.transpose(kernels,(2,0,3,1,4))
    kernels = tf.reshape(kernels,[square_size*iy,square_size*ix,3])
    kernels = tf.pad(kernels, [[0,pad],[0,pad],[0,0]], mode='constant',constant_values=black)
    return tf.expand_dims(kernels, axis=0)

def put_grads_on_grid(grads):
    b_size=tf.shape(grads, out_type=tf.int32)[0]
    iy=tf.shape(grads, out_type=tf.int32)[1]
    ix=tf.shape(grads, out_type=tf.int32)[2]
    n_ch=tf.cast(tf.shape(grads, out_type=tf.int32)[4], tf.float32)
    square_size=tf.cast(tf.ceil(tf.sqrt(n_ch)),tf.float32)
    z_pad=tf.cast(square_size**2-n_ch, tf.int32)
    square_size=tf.cast(square_size,tf.int32)
    black=tf.minimum(0.0,tf.reduce_min(grads))
    pad=1+tf.cast((ix/64), tf.int32)
    grads = tf.pad(grads, [[0,0],[0,0],[0,0],[0,0],[0,z_pad]], mode='constant',constant_values=black)
    grads = tf.transpose(grads,(0,1,2,4,3))
    grads = tf.reshape(grads,[b_size,iy,ix,square_size,square_size,3])
    grads = tf.pad(grads, [[0,0],[pad,0],[pad,0],[0,0],[0,0],[0,0]], mode='constant',constant_values=black)
    iy+=pad
    ix+=pad
    grads = tf.transpose(grads,(0,3,1,4,2,5))
    grads = tf.reshape(grads,[b_size,square_size*iy,square_size*ix,3])
    return tf.pad(grads, [[0,0],[0,pad],[0,pad],[0,0]], mode='constant',constant_values=black)

#def deconvolution(x, feedDict, ft_maps, features_list, batch_size, input_size):
#    deconv=[]
#    sess=tf.get_default_session()
#    for ft_map, key in zip(ft_maps, features_list):
#        ft_shape=ft_map.get_shape()
#        img_shape=input_size
#        ft_deconv=np.empty((batch_size,)+img_shape+(ft_shape[3],))
#        for ch in xrange(ft_shape[3]):
#            score = tf.reduce_mean(ft_map[:,:,:,ch])
#            grad = tf.gradients(score, x)[0]
#            grad=normalize_std(grad)
#            deconv_op=tf.multiply(x,grad)
#            ft_deconv[:,:,:,:,ch]=sess.run(grad, feed_dict=feedDict)
#        deconv.append(ft_deconv)
#    return deconv
