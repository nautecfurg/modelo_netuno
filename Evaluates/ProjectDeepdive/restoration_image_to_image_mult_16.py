import glob
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image
import math
import evaluate


class RestorationImageToImageMult16(evaluate.Evaluate):
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
                glob.glob(os.path.join(evaluate_input_dir, "*.bmp"))
        #reuse = None
        
        for name in im_names:
            with tf.Graph().as_default():
                image = Image.open(name).convert('RGB')
                image = np.array(image, dtype=np.float32) / 255.0
                architecture_instance.input_size=image.shape[0:2]
                image.shape = (1,)+ image.shape
                image_shape = image.shape
                architecture_input = tf.placeholder("float", shape=image.shape,
                                                name="input_image")

                height = image_shape[1]
                width = image_shape[2]

                if height % 16 !=0:
                    height += 16 - (height % 16)
                if width % 16 !=0:
                    width += 16 - (width % 16)

                architecture_input2 = tf.image.resize_images(architecture_input, [height, width])
                with tf.variable_scope("model/architecture", reuse=None):
                    architecture_output = architecture_instance.prediction(architecture_input2,
                                                                        training=False)
                architecture_output = tf.image.resize_images(architecture_output, image_shape[1:3])
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
                print(model_file_path)
                print(tf.contrib.framework.list_variables(model_file_path))
                print(saver.restore(sess, model_file_path))
                
                print("Model restored.")

                feed_dict = {architecture_input: image}
                output_np = sess.run(architecture_output, feed_dict=feed_dict)
                
                output_np.shape = output_np.shape[1:]
                output_np = (output_np * 255).astype(np.uint8)
                output = Image.fromarray(output_np)
                name = name[len(evaluate_input_dir):]
                output.save(evaluate_output_path + name)
                sess.close()