import glob
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image
import math
import evaluate


class RestorationTransmissionDehazenet(evaluate.Evaluate):
    def eval(self, opt_values, architecture_instance):
        execution_dir = opt_values["execution_path"]
        evaluate_input_dir = opt_values["evaluate_path"]

        evaluate_output_dir = os.path.join(execution_dir, "Evaluates")
        time_str = time.strftime("%Y-%m-%d_%H:%M/")
        evaluate_output_name = "Output" + time_str

        evaluate_output_path = os.path.join(evaluate_output_dir, evaluate_output_name)
        os.makedirs(evaluate_output_path)
        model_dir = os.path.join(execution_dir, "Model")
        im_names = glob.glob(os.path.join(evaluate_input_dir, "*.jpg")) +\
                glob.glob(os.path.join(evaluate_input_dir, "*.png")) +\
                glob.glob(os.path.join(evaluate_input_dir, "*.bmp "))
        #reuse = None

        for name in im_names:
            with tf.Graph().as_default():
                image = Image.open(name).convert('RGB')
                image = np.array(image, dtype=np.float32) / 255.0
                architecture_instance.input_size=image.shape[0:2]
                image.shape = (1,)+ image.shape
                architecture_input = tf.placeholder("float", shape=image.shape,
                                                name="input_image")
                
                with tf.variable_scope("model", reuse=None):
                    architecture_output = architecture_instance.prediction(architecture_input,
                                                                        training=False)

                # The op for initializing the variables.
                init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
                # Add op to restore all the variables.
                saver = tf.train.Saver()
                # Create a session for running operations in the Graph.
                sess = tf.Session()
                # Initialize the variables
                sess.run(init_op)
                # Restore variables from disk.
                model_file_path = os.path.join(model_dir, "model.ckpt")
                saver.restore(sess, model_file_path)
                print("Model restored.")

                feed_dict = {architecture_input: image}
                output_np = sess.run(architecture_output, feed_dict=feed_dict)
                
                output_np.shape = output_np.shape[1:-1]

                image.shape = image.shape[1:]
                A = AtmLight(image, output_np)            #ambient light estimation
                restored_output_np = Recover(image, output_np, A, 0)
                restored_output_np = (restored_output_np * 255).astype(np.uint8)
                restored_output = Image.fromarray(restored_output_np)
                name = name[len(evaluate_input_dir):]
                restored_output.save(evaluate_output_path + name)
                output_np = np.clip(output_np,0,1)
                output_np = (output_np * 255).astype(np.uint8)
                output = Image.fromarray(output_np, mode='L')
                output.save(evaluate_output_path + name+"_t.jpg")
                sess.close()

def AtmLight(im, dark):
    [h,w] = im.shape[0:2]
    print ([h,w])
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1)
    imvec = im.reshape(imsz,3)
    indices = np.argsort(darkvec,axis=0)
    indices = indices[:numpx]
    A = np.zeros([3])
    atmsum = np.zeros([3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx
    A = A[0]
    return A

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape,im.dtype)
    t = np.clip(t,tx,1) 
    #t = np.maximum(t, tx)
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[ind]*(1 - t))/t 
    res = np.clip(res - np.amin(res), a_min=0,  a_max=None)
    res = np.clip(res / np.amax(res), a_min=None, a_max=1)
    return res