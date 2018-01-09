"""Aplica a TransmissionNet patch a patch em imagens de teste"""
import os
import glob
import time

import utils
import dataset
import numpy as np
import tensorflow as tf

from PIL import Image

from Evaluates import simulator
import evaluate

class EvaluateTransmissionNet(evaluate.Evaluate):
    def __init__(self, batch_size=256):
        self.batch_size = batch_size
        self.dataset = utils.get_implementation(dataset.Dataset, 'DatasetTransmission')
        self.patch_size = self.dataset.patch_size
    def eval(self, opt_values, architecture_instance):
        patch_size = self.patch_size
        execution_dir = opt_values["execution_path"]
        input_dir = opt_values["evaluate_path"]

        output_dir = os.path.join(execution_dir, "Evaluates")
        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
        output_name = "Output" + time_str
        output_path = os.path.join(output_dir, output_name)
        os.makedirs(output_path)
        model_dir = os.path.join(execution_dir, "Model/")
        im_names = glob.glob(os.path.join(input_dir, "*color*.jpg")) +\
                glob.glob(os.path.join(input_dir, "*color*.png"))
        depth_names = glob.glob(os.path.join(input_dir, "*Depth*.jpg")) +\
                glob.glob(os.path.join(input_dir, "*Depth*.png"))
        if im_names == []:
            return
        im_names.sort()
        depth_names.sort()
        img = Image.open(im_names[0]).convert('RGB')
        img_shape = np.shape(img) #supoe-se que todas as imagens tem o mesmo shape
        num_imgs = len(im_names)
        width = img_shape[0]
        height = img_shape[1]
        width_out = width - patch_size[0] + 1
        height_out = height - patch_size[1] + 1
        pixel_vector = np.zeros([], dtype=np.float32)

        with tf.Graph().as_default():

            batch = np.zeros([self.batch_size,] + patch_size, dtype=np.float32) 
            current_batch_size = 0 #contador de tamanho pra quando chegar no final das imagens

            architecture_input = tf.placeholder("float", shape=[self.batch_size, ] + patch_size,
                                                    name="input_image")
            with tf.variable_scope('model'):
                architecture_output = architecture_instance.prediction(architecture_input,
                                                                       training=False)

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(init_op)
            model_file_path = os.path.join(model_dir, "model.ckpt")
            saver.restore(sess, model_file_path)
            print("Model restored.")
            num_batches = 0
            total_batches = width_out * height_out * num_imgs // self.batch_size
            current_time = time.time()

            #TODO: config???
            turbidity_path = "Datasets/Data/TurbidityAzul"
            turbidity_size = (128, 128)
            batch_size = 1
            for count, img_name in enumerate(im_names):
                c, binf, ranges = simulator.acquireProperties(turbidity_path, turbidity_size, batch_size, .45, .55, sess)
                img = Image.open(img_name).convert('RGB')
                img = np.array(img, dtype=np.float32) / 255.0
                depth = Image.open(depth_names[count])
                depth = np.array(depth, dtype=np.float32) / 12000.0
                tf_img = tf.convert_to_tensor(img)
                tf_depth = tf.convert_to_tensor(depth)
                tf_depth = tf.stack([tf_depth, tf_depth, tf_depth], axis=2)
                img = simulator.applyTurbidity(tf_img, tf_depth, c, binf, ranges)
                img = sess.run(img)
                img = np.reshape(img, img_shape)
                for y in range(height_out):
                    for x in range(width_out):
                        batch[current_batch_size,:,:,:] = img[x:x+patch_size[0], y:y+patch_size[0],:]
                        current_batch_size += 1
                        if current_batch_size == self.batch_size or\
                                (count == num_imgs - 1 and x == width_out - 1 and\
                                y == height_out - 1):
                            feed_dict = {architecture_input: batch}
                            output_np = sess.run(architecture_output, feed_dict=feed_dict)
                            pixel_vector = np.append(pixel_vector, output_np)
                            current_batch_size = 0
                            if(num_batches % 1000 == 0 or\
                                    (count == num_imgs - 1 and x == width_out - 1 and\
                                    y == height_out - 1)):
                                time_diff = time.time() - current_time
                                current_time = time.time()
                                print("consuming batch %d out of %d (%.3f s)" % (num_batches, total_batches, time_diff))
                            num_batches += 1
            for i in range(0, num_imgs * width_out * height_out - 1, width_out * height_out):
                out_img_np = np.reshape(pixel_vector[i:i + width_out * height_out],\
                        (width_out, height_out))
                out_img_np = (out_img_np * 255).astype(np.uint8)
                out_img_np = np.rot90(out_img_np, k=3)
                out_img_np = np.fliplr(out_img_np)
                output = Image.fromarray(out_img_np, mode='L')
                input_name = os.path.split(im_names[(i // (width_out * height_out))])[1]
                name = input_name[:-3] + "_transmission.png"
                output.save(os.path.join(output_path, name))
