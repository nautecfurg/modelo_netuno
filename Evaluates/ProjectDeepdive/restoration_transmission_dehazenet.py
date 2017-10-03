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
        time_str = time.strftime("%Y-%m-%d_%H:%M")
        evaluate_output_name = "Output" + time_str

        evaluate_output_path = os.path.join(evaluate_output_dir, evaluate_output_name)
        os.makedirs(evaluate_output_path)
        model_dir = os.path.join(execution_dir, "model")
        im_names = glob.glob(os.path.join(evaluate_input_dir, "*.jpg")) +\
                glob.glob(os.path.join(evaluate_input_dir, "*.png"))

        with tf.Graph().as_default():

            architecture_input = tf.placeholder("float", shape=(None, None, None, 3),
                                                name="input_image")
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

            for name in im_names:
                image = Image.open(name).convert('RGB')
                image = np.array(image, dtype=np.float32) / 255.0
                image_orig = image
                image.shape = (1,)+ image.shape
                feed_dict = {architecture_input: image}
                output_np = sess.run(architecture_output, feed_dict=feed_dict)
                output_np.shape = output_np.shape[1:-1]
                A = AtmLight(image_orig, output_np)            #ambient light estimation
                restored_output_np = Recover(image_orig, output_np, A, 0.1)
                restored_output_np = (output_np * 255).astype(np.uint8)
                print(restored_output_np.shape)
                restored_output = Image.fromarray(restored_output_np)
                name = name[len(evaluate_input_dir):]
                restored_output.save(evaluate_output_path + name)

def AtmLight(im, dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1)
    imvec = im.reshape(imsz,3)
    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]
    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx
    return A
def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape,im.dtype)
    t = np.maximum(t, tx)
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
    return res