import glob
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image

import evaluate


class Evaluate_tubid_image_depth(evaluate.Evaluate):
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
                image.shape = (1,)+ image.shape
                feed_dict = {architecture_input: image}
                output_np = sess.run(architecture_output, feed_dict=feed_dict)
                output_np.shape = output_np.shape[1:-1]
                output_np = (output_np * 255).astype(np.uint8)
                print(output_np.shape)
                output = Image.fromarray(output_np, mode='L')
                name = name[len(evaluate_input_dir):]
                output.save(evaluate_output_path + name)
