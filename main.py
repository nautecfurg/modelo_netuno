import getopt
from PIL import Image
import glob
import json
import os
import sys
import time
import re

import numpy as np
import tensorflow as tf

import architecture
import Architectures
import evaluate
import Evaluates
import dataset
import dataset_manager
import DatasetManagers
import Datasets
import loss
import Losses
import optimizer
import Optimizers
import utils
from visualization_utils import *
from activation_maximization import maximize_activation

HELP_MSG = """usage: python main.py [-h] [-m MODE] [-a ARCHITECTURE] [-d DATASET]
           [-g DATASET_MANAGER] [-l LOSS] [-o OPTIMIZER] [-e EVALUATE] [-p EXECUTION_PATH]
           [--evaluate_path EVALUATE_PATH] [--visualize_layers "KEY1,KEY2,...,KEYN"]
optional arguments
    -h, --help                              show this help message and exit
    -m, --mode MODE                         specifies one of the possible modes (train,
                                            evaluate, restore, dataset_manage, visualize)
    -a, --architecture ARCHITECTURE         specifies an architecture it is required in all
                                            modes except for dataset_manage
    -d, --dataset DATASET                   specifies one dataset implementation it is
                                            required for all modes except dataset_manage  
    -g, --dataset_manager DATASET_MANAGER   specifies one dataset manager implementation
                                            it is required and exclusively used in 
                                            dataset_manage mode
    -l, --loss LOSS                         specifies one loss implementation and is
                                            required for all modes except dataset_manage
    -o, --optimizer OPTIMIZER               specifies an optimizer implementation and is
                                            required for train and restore modes
    -e, --evaluate EVALUATE                 specifies an evaluate implementation it is
                                            required for evaluate mode only
    --evaluate_path EVALUATE_PATH           specifies the folder were the exemples to be evaluated
                                            are located
    -p, --execution_path EXECUTION_PATH     specifies an execution path and it is required
                                            only in restore and visualize modes
    --visualize_layers "KEY1,KEY2,...,KEYN" a list of layers to be visualized, used only
                                            in visualize mode
"""



def process_args(argv):
    """
    It checks and organizes the arguments in a dictionary.

    Args:
        argv: list containing all parameters and arguments.

    Returns:
        opt_values: dictionary containing parameters as keys and arguments as values.
    """

    try:
        long_opts = ["help", "architecture=", "dataset=",
                     "dataset_manager=", "loss=", "execution_path=",
                     "optimizer=", "evaluate_path=", "evaluate=", "visualize_layers="]
        opts, _ = getopt.getopt(argv, "ha:d:m:g:l:p:o:e", long_opts)
        if opts == []:
            print(HELP_MSG)
            sys.exit(2)
    except getopt.GetoptError:
        print(HELP_MSG)
        sys.exit(2)
    opt_values = {}
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(HELP_MSG)
            sys.exit()
        elif opt in ("-m", "--mode"):
            if arg in ('train', 'evaluate', 'restore', 'dataset_manage', 'visualize', 'activation_maximization'):
                opt_values['execution_mode'] = arg
            else:
                print(arg + ' is not a possible execution mode')

                sys.exit(2)
        elif opt in ("-a", "--architecture"):
            opt_values['architecture_name'] = \
                                utils.arg_validation(arg, architecture.Architecture)
        elif opt in ("-d", "--dataset"):
            opt_values['dataset_name'] = utils.arg_validation(arg, dataset.Dataset)
        elif opt in ("-g", "--dataset_manager"):
            opt_values['dataset_manager_name'] = \
                        utils.arg_validation(arg, dataset_manager.DatasetManager)
        elif opt in ("-e", "--evaluate"):
            opt_values['evaluate_name'] = utils.arg_validation(arg, evaluate.Evaluate)
        elif opt in ("-l", "--loss_name"):
            opt_values['loss_name'] = \
                        utils.arg_validation(arg, loss.Loss)
        elif opt in ("-p", "--execution_path"):
            if os.path.isdir(arg):
                opt_values['execution_path'] = arg
            else:
                print(arg + " is not a valid folder path.")
                sys.exit(2)
        elif opt == "--evaluate_path":
            if os.path.isdir(arg):
                opt_values['evaluate_path'] = arg
            else:
                print(arg + " is not a valid folder path.")
                sys.exit(2)
        elif opt in ("-o", "--optimizer"):
            opt_values['optimizer_name'] = \
                        utils.arg_validation(arg, optimizer.Optimizer)
        elif opt == "--visualize_layers":
            opt_values['visualize_layers'] = arg


    check_needed_parameters(opt_values)
    return opt_values

def check_needed_parameters(parameter_values):
    """This function checks if the needed are given in the main call.

    According with the execution mode this function checks if the needed
    parameteres were defined. If one parameter is missing, this function
    gives an output with the missing parameter and the chosen mode.

    Args:
        parameter_values: Dictionary containing all paramenter names and arguments.
    """
    if parameter_values["execution_mode"] == "train":
        needed_parameters = ["architecture_name", "dataset_name", "loss_name", "optimizer_name"]
    elif parameter_values["execution_mode"] == "restore":
        needed_parameters = ["architecture_name", "dataset_name", "loss_name", "optimizer_name",
                             "execution_path"]
    elif parameter_values["execution_mode"] == "evaluate":
        needed_parameters = ["architecture_name", "execution_path", "evaluate_path",
                             "evaluate_name"]
    elif parameter_values["execution_mode"] == "dataset_manage":
        needed_parameters = ["dataset_manager_name"]
    elif parameter_values["execution_mode"] == "visualize":
        needed_parameters = ["architecture_name", "dataset_name", "execution_path",
                             "visualize_layers"]
    elif parameter_values["execution_mode"] == "activation_maximization":
        needed_parameters = ["architecture_name", "execution_path",
                             "visualize_layers"]

    else:
        print("Parameters list must contain an execution_mode.")
        sys.exit(2)

    for parameter in needed_parameters:
        if parameter not in parameter_values:
            print("Parameters list must contain an " + parameter + " in the " +\
                  parameter_values["execution_mode"]+" mode.")
            sys.exit(2)

def get_tensorboard_command(train=None, test=None, visualize=None):
    """This function creates a command that can be executed in the terminal 
    to run tensorboard in the current training and test directories
    Args:
        train_dir: String with the path to the training summaries
        test_dir: String with the path to the test summaries"""

    if (train is not None) and (test is not None):
        train_string="train:"+os.path.abspath(train)
        test_string="test:"+os.path.abspath(test)
        return "tensorboard --logdir "+train_string+","+test_string
    elif visualize is not None:
        return "tensorboard --logdir visualize:"+visualize

def load_actv_max_params():
    """This function reads the activation maximization parameters from activation_maximization_config.json.
    If the file or some of its parameters are missing, default values will be used.
    """
    parameters = {}
    if os.path.exists('activation_maximization_config.json'):
      print("Activation maximization configuration file found.")
      data = json.load(open('activation_maximization_config.json'))      
      try:
        actv_max_input_size_str=data["actv_max_input_size"]
        w, h, ch = re.split(",",actv_max_input_size_str)
        parameters['actv_max_input_size']=(int(w), int(h), int(ch))
      except KeyError:
        parameters['actv_max_input_size']=(224, 224, 3)
      try:
        parameters['noise']=data["noise"]
      except KeyError:
        parameters['noise']=True
      try:
        parameters['step_size']=float(data["step_size"])
      except KeyError:
        parameters['step_size']=1
      try:
        parameters['actv_max_iters']=int(data["actv_max_iters"])
      except KeyError:
        parameters['actv_max_iters']=100
      try:
        parameters['blur_every']=int(data["blur_every"])
      except KeyError:
        parameters['blur_every']=4
      try:
        parameters['blur_width']=int(data["blur_width"])
      except KeyError:
        parameters['blur_width']=1
      try:
        parameters['lap_norm_levels']=int(data["lap_norm_levels"])
      except KeyError:
        parameters['lap_norm_levels']=5
      try:
        parameters['tv_lambda']=float(data["tv_lambda"])
      except KeyError:
        parameters['tv_lambda']=0
      try:
        parameters['tv_beta']=float(data["tv_beta"])
      except KeyError:
        parameters['tv_beta']=2.0
      try:
        parameters['jitter']=data["jitter"]
      except KeyError:
        parameters['jitter']=False
      try:
        parameters['scale']=data["scale"]
      except KeyError:
        parameters['scale']=False 
      try:
        parameters['rotate']=data["rotate"]
      except KeyError:
        parameters['rotate']=False    

    else:
      print("Activation maximization configuration file not found.")
      print("Using default values.")
      parameters['actv_max_input_size']=(224, 224, 3)
      parameters['noise']=True
      parameters['step_size']=1
      parameters['actv_max_iters']=100
      parameters['blur_every']=4
      parameters['blur_width']=1
      parameters['lap_norm_levels']=5
      parameters['tv_lambda']=0
      parameters['tv_beta']=2.0
      parameters['jitter']=False
      parameters['scale']=False
      parameters['rotate']=False
    return parameters

def training(loss_op, optimizer_imp):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        loss_op: Loss tensor, from loss().
        optimizer_imp: optmizer instance.

    Returns:
        train_op: The Op for training.
    """
    # Get variable list
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/architecture")
    
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('train_loss', loss_op)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer_imp.minimize(loss_op, global_step=global_step, var_list=model_vars)
    return train_op, global_step


def run_training(opt_values):
    """
    Runs the training

    This function is responsible for instanciating dataset, architecture and loss.abs

    Args:
        opt_values: dictionary containing parameters as keys and arguments as values.

    """
    # Get architecture, dataset and loss name
    arch_name = opt_values['architecture_name']
    dataset_name = opt_values['dataset_name']
    loss_name = opt_values['loss_name']
    optimizer_name = opt_values['optimizer_name']

    time_str = time.strftime("%Y-%m-%d_%H:%M")
    
    if opt_values["execution_mode"] == "train":
        execution_dir = "Executions/" + dataset_name + "_" + arch_name + "_" + loss_name +\
                    "_" + time_str
        os.makedirs(execution_dir)
    elif opt_values["execution_mode"] == "restore":
        execution_dir = opt_values["execution_path"]
    log(opt_values, execution_dir)
    # Create summary
    model_dir = os.path.join(execution_dir, "Model")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    summary_dir = os.path.join(execution_dir, "Summary")
    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)


    # Get implementations
    architecture_imp = utils.get_implementation(architecture.Architecture, arch_name)
    dataset_imp = utils.get_implementation(dataset.Dataset, dataset_name)
    loss_imp = utils.get_implementation(loss.Loss, loss_name)
    optimizer_imp = utils.get_implementation(optimizer.Optimizer, optimizer_name)

    # Tell TensorFlow that the model will be built into the default Graph.
    graph = tf.Graph()
    with graph.as_default():
        # if it load step to continue on the same point on dataset
        execution_mode = opt_values["execution_mode"]
        if execution_mode == "train":
            initial_step = 0
        else:
            initial_step = tf.train.load_variable(model_dir, "global_step")
        # Input and target output pairs.
        architecture_input, target_output = dataset_imp.next_batch_train(initial_step)

        with tf.variable_scope("model"):
            with tf.variable_scope("architecture"):
                architecture_output = architecture_imp.prediction(architecture_input, training=True)
            loss_op = loss_imp.evaluate(architecture_input, architecture_output, target_output)
        train_op, global_step = training(loss_op, optimizer_imp)

        if loss_imp.trainable():
            loss_tr = loss_imp.train(optimizer_imp)
        # Merge all train summaries and write
        merged = tf.summary.merge_all()
        # Test
        architecture_input_test, target_output_test, init = dataset_imp.next_batch_test()

        with tf.variable_scope("model", reuse=True):
            with tf.variable_scope("architecture", reuse=True):
                architecture_output_test = architecture_imp.prediction(architecture_input_test,
                                                                   training=False) # TODO: false?
            loss_op_test = loss_imp.evaluate(architecture_input_test, architecture_output_test, target_output_test)
        tf_test_loss = tf.placeholder(tf.float32, shape=(), name="tf_test_loss")
        test_loss = tf.summary.scalar('test_loss', tf_test_loss)

        train_summary_dir=os.path.join(summary_dir, "Train")
        test_summary_dir=os.path.join(summary_dir, "Test")
        train_writer = tf.summary.FileWriter(train_summary_dir)
        test_writer = tf.summary.FileWriter(test_summary_dir)
        # # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Create a session for running operations in the Graph.
        sess = tf.Session()
        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)
        sess.run(init)

        if execution_mode == "restore":
            # Restore variables from disk.
            model_file_path = os.path.join(model_dir, "model.ckpt")
            saver.restore(sess, model_file_path)
            print("Model restored.")

        tensorboard_command=get_tensorboard_command(train_summary_dir, test_summary_dir)
        print("To run tensorboard, execute the following command in the terminal:")
        print(tensorboard_command)
        step = sess.run(global_step)

        try:
            while True:
                start_time = time.time()
                sess.run(init)
                # Run one step of the model.  The return values are
                # the activations from the `train_op` (which is
                # discarded) and the `loss` op.  To inspect the values
                # of your ops or variables, you may include them in
                # the list passed to sess.run() and the value tensors
                # will be returned in the tuple from the call.

                # Warning: Calling the sess.run will advance the dataset
                # to the next batch.
                if step % architecture_imp.get_summary_writing_period() == 0:
                    # Train Discriminator & Generator
                    loss_value, _, _, summary = sess.run([loss_op, loss_tr, train_op, merged])
                else:
                    # Train Discriminator & Generator
                    loss_value, _, _ = sess.run([loss_op, loss_tr, train_op])
                duration = time.time() - start_time
                if step % architecture_imp.get_summary_writing_period() == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, np.mean(loss_value),
                                                               duration))
                    train_writer.add_summary(summary, step)

                if step % architecture_imp.get_validation_period() == 0:
                    loss_value_sum = 0.0
                    count_test = 0.0
                    try:
                        start_time = time.time()
                        while True:
                            loss_value_test = sess.run(loss_op_test)
                            count_test = count_test + 1
                            loss_value_sum = loss_value_sum + loss_value_test
                    except tf.errors.OutOfRangeError:
                        duration_test = time.time() - start_time
                        print('Done testing. (%.3f sec)' % (duration_test))
                    loss_value_test = loss_value_sum / count_test
                    summary_test = sess.run(test_loss,
                                            feed_dict={tf_test_loss:loss_value_test})
                    test_writer.add_summary(summary_test, step)
                if step % architecture_imp.get_model_saving_period() == 0:
                    # Save the variables to disk.
                    save_path = saver.save(sess, model_dir + "/model.ckpt")
                    print("Model saved in file: %s" % save_path)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training, %d steps.' % (step))
        finally:
            sess.close()


def log(opt_values, execution_dir): #maybe in another module?
    """
    Stores a .json logfile in execution_dir/logs,
    based on the execution options opt_values.

    Args:
        opt_values: dictionary with the command line arguments of main.py.
        execution_dir: the directory regarding the execution to log.

    Returns:
        Nothing.
    """
    # Get architecture, dataset and loss name
    architecture_name = opt_values['architecture_name']
    dataset_name = opt_values['dataset_name']
    loss_name = opt_values['loss_name']
    optimizer_name = opt_values['optimizer_name']
    # Get implementations
    architecture_imp = utils.get_implementation(architecture.Architecture, architecture_name)
    dataset_imp = utils.get_implementation(dataset.Dataset, dataset_name)
    loss_imp = utils.get_implementation(loss.Loss, loss_name)
    optimizer_imp = utils.get_implementation(optimizer.Optimizer, optimizer_name)

    today = time.strftime("%Y-%m-%d_%H:%M")
    log_name = dataset_name + "_" + architecture_name + "_" + loss_name + "_" +\
               opt_values['execution_mode'] + "_" + today +".json"
    json_data = {}
    if hasattr(architecture_imp, 'config_dict'):
        json_data["architecture"] = architecture_imp.config_dict
    if hasattr(loss_imp, 'config_dict'):
        json_data["loss"] = loss_imp.config_dict
    if hasattr(dataset_imp, 'config_dict'):
        json_data["dataset"] = dataset_imp.config_dict
    if hasattr(optimizer_imp, 'config_dict'):
        json_data["optimizer"] = optimizer_imp.config_dict
    json_data["execution_mode"] = opt_values['execution_mode']
    json_data["architecture_name"] = architecture_name
    json_data["dataset_name"] = dataset_name
    json_data["loss_name"] = loss_name
    json_data["optimizer_name"] = optimizer_name

    log_dir = os.path.join(execution_dir, "Logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_name)
    with open(log_path, 'w') as outfile:
        json.dump(json_data, outfile)

def run_evaluate(opt_values):
    """
    This function performs the evaluation.
    """
    eval_name = opt_values['evaluate_name']
    evaluate_imp = utils.get_implementation(evaluate.Evaluate, eval_name)
    arch_name = opt_values['architecture_name']
    architecture_imp = utils.get_implementation(architecture.Architecture, arch_name)
    evaluate_imp.eval(opt_values, architecture_imp)

def run_dataset_manager(opt_values):
    dm_name = opt_values['dataset_manager_name']
    dataset_manager_imp = utils.get_implementation(dataset_manager.DatasetManager, dm_name)
    dataset_manager_imp.convert_data()

def run_visualization(opt_values):
    """
    Runs the visualization

    This function is responsible for instanciating dataset, architecture

    Args:
        opt_values: dictionary containing parameters as keys and arguments as values.

    """
    # Get architecture, dataset and loss name
    arch_name = opt_values['architecture_name']
    dataset_name = opt_values['dataset_name']
    
    execution_dir = opt_values["execution_path"]
    model_dir = os.path.join(execution_dir, "Model")

    summary_dir = os.path.join(execution_dir, "Summary")
    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)

    # Get implementations
    architecture_imp = utils.get_implementation(architecture.Architecture, arch_name)
    dataset_imp = utils.get_implementation(dataset.Dataset, dataset_name)

    # Tell TensorFlow that the model will be built into the default Graph.
    graph = tf.Graph()
    with graph.as_default():
        # Input and target output pairs.
        architecture_input, target_output = dataset_imp.next_batch_train(0)

        with tf.variable_scope("model"):
            architecture_output = architecture_imp.prediction(architecture_input, training=False)

        visualize_summary_dir=os.path.join(summary_dir, "Visualize_"+dataset_name)
        visualize_writer = tf.summary.FileWriter(visualize_summary_dir)

        # # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Add ops to save and restore all the variables.
        sess = tf.InteractiveSession()

        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)

        # Restore variables from disk.
        model_file_path = os.path.join(model_dir, "model.ckpt")
        saver.restore(sess, model_file_path)
        print("Model restored.")

        tensorboard_command=get_tensorboard_command(visualize=visualize_summary_dir)
        print("To run tensorboard, execute the following command in the terminal:")
        print(tensorboard_command)
        step=0

        layer_summaries=[]
        layer_avg_ops=[]
        layer_avgs={}
        json_name="average_activations.json"
        json_file_path = os.path.join(visualize_summary_dir, json_name)
        if os.path.isfile(json_file_path):
            outfile= open(json_file_path,'r+')
            layer_avgs=json.load(outfile)
            outfile.close()

        key_list=re.split("[,; ]",opt_values['visualize_layers'])
        for k in key_list:
            layer=architecture_imp.get_layer(k)           
            layer_grid=put_features_on_grid(layer)
            layer_summaries.append(tf.summary.image(k, layer_grid, max_outputs=512))
            layer_avgs[k]=[]
            layer_avg_ops.append(tf.reduce_mean(layer,axis=(1,2)))

        try:

            while True:
                summaries = sess.run(layer_summaries)
                batch_avgs = sess.run(layer_avg_ops)
                for k, avg in zip(key_list, batch_avgs):
                    layer_avgs[k].extend(avg.tolist())
                for summary in summaries:
                    visualize_writer.add_summary(summary, step)
                step+=1
        except tf.errors.OutOfRangeError:
            print('Done visualizing, %d steps.' % (step))
        finally:
            sess.close()                       
            with open(json_file_path, 'w') as outfile:
                json.dump(layer_avgs, outfile)
            
def run_activation_maximization(opt_values):
    """
    Runs the activation maximization of network layers specified by the --visualize_layers parameter

    This function is responsible for instanciating architecture

    Args:
        opt_values: dictionary containing parameters as keys and arguments as values.

    """
    # Get architecture name
    arch_name = opt_values['architecture_name']
    
    execution_dir = opt_values["execution_path"]
    model_dir = os.path.join(execution_dir, "Model")

    summary_dir = os.path.join(execution_dir, "Summary")
    if not os.path.isdir(summary_dir):
        os.makedirs(summary_dir)

    # Read activation maximization parameters from configuration file.
    actv_max_params=load_actv_max_params()
    # This is the size of the input generated by activation maximization.
    actv_max_input_size=actv_max_params['actv_max_input_size']
    # True for random image initialization, false for gray image.
    noise=actv_max_params['noise']
    # Activation maximization gradient step size.
    step_size=actv_max_params['step_size']
    # Number of activation maximization itertions.
    actv_max_iters=actv_max_params['actv_max_iters']
    # Frequency of the blurring operation. 0 means no bluring. 
    blur_every=actv_max_params['blur_every']
    # Width of the blurring kernel.
    blur_width=actv_max_params['blur_width']
    # Number of levels used in laplacian pyramid gradient normalization.
    # A 1 level pyramid is equivalent to no normalization.
    lap_norm_levels=actv_max_params['lap_norm_levels']
    # Controls the influence of the total variation norm in the optimization.
    tv_lambda=actv_max_params['tv_lambda']
    # The exponent used to calculate the total variation norm.
    tv_beta=actv_max_params['tv_lambda']
    # True for jitter (translation) regularization
    jitter=actv_max_params['jitter']
    # True for scale regularization
    scale=actv_max_params['scale']
    # True for rotation regularization
    rotate=actv_max_params['rotate']

    # Get implementations
    architecture_imp = utils.get_implementation(architecture.Architecture, arch_name)

    # Tell TensorFlow that the model will be built into the default Graph.
    graph = tf.Graph()
    with graph.as_default():
        architecture_input = tf.placeholder(tf.float32, shape=(1,)+actv_max_input_size)

        with tf.variable_scope("model"):
            architecture_output = architecture_imp.prediction(architecture_input, training=False)

        # Generates the name of the folder where the tensorboard summaries will be saved.
        summary_name="ActivationMaximization"
        if(noise):
          summary_name+="Noise"
        else:
          summary_name+="NoNoise"
        summary_name+="Step"+str(step_size)
        summary_name+="Iters"+str(actv_max_iters)
        summary_name+="BlurEvery"+str(blur_every)
        summary_name+="BlurWidth"+str(blur_width)
        summary_name+="LapNorm"+str(lap_norm_levels)
        summary_name+="TVlambda"+str(tv_lambda)
        summary_name+="TVbeta"+str(tv_beta)
        if(jitter):
          summary_name+="Jitter"
        if(scale):
          summary_name+="Scaling"
        if(rotate):
          summary_name+="Rotation"



        visualize_summary_dir=os.path.join(summary_dir, summary_name)
        visualize_writer = tf.summary.FileWriter(visualize_summary_dir)

        # # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Add ops to save and restore all the variables.
        sess = tf.InteractiveSession()

        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)

        # Restore variables from disk.
        model_file_path = os.path.join(model_dir, "model.ckpt")
        saver.restore(sess, model_file_path)
        print("Model restored.")

        tensorboard_command=get_tensorboard_command(visualize=visualize_summary_dir)
        print("To run tensorboard, execute the following command in the terminal:")
        print(tensorboard_command)

        key_list=re.split("[,; ]",opt_values['visualize_layers'])

        try:
            print("Running Activation Maximization")
            # Every layer specified with --visualize_layers.
            for key in key_list:
              print("Layer "+key)	
              ft=architecture_imp.get_layer(key)
              n_channels=ft.get_shape()[3]
              # Initializes the grid.
              opt_grid=np.empty((1,)+actv_max_input_size+(n_channels,))
              # Every channel.
              for ch in range(n_channels):
                print("Channel "+str(ch))
                opt_output=maximize_activation(actv_max_input_size, architecture_input,ft[:,:,:,ch],noise,step_size,
                                               actv_max_iters,blur_every,blur_width,lap_norm_levels,tv_lambda,tv_beta,
                                               jitter, scale, rotate)
                opt_output -= opt_output.min()
                opt_output *= (255/(opt_output.max()+0.0001))
                opt_grid[0,:,:,:,ch]=opt_output
              # Saves a grid containing the results of all channels of a layers in the summaries.
              opt_grid_name=key+"_maximization"
              opt_grid_img=put_grads_on_grid(opt_grid.astype(np.float32))
              opt_grid_summary=tf.summary.image(opt_grid_name, opt_grid_img)
              opt_grid_summary_str=sess.run(opt_grid_summary)
              visualize_writer.add_summary(opt_grid_summary_str,0)

        finally:
            sess.close()                       

        

def main(opt_values):
    """
    Main function.
    """
    execution_mode = opt_values['execution_mode']
    if execution_mode in ('train', 'restore'):
        run_training(opt_values)
    elif execution_mode == 'evaluate':
        run_evaluate(opt_values)
    elif execution_mode == 'dataset_manage':
        run_dataset_manager(opt_values)
    elif execution_mode == 'visualize':
        run_visualization(opt_values)
    elif execution_mode == 'activation_maximization':
        run_activation_maximization(opt_values)

if __name__ == "__main__":
    OPT_VALUES = process_args(sys.argv[1:])
    main(OPT_VALUES)
