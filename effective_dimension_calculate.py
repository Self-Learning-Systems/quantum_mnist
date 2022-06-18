"""

Author: Maniraman Periyasamy
Organization: Maniraman Periyasamy

This file calculates the effective dimension of a given circuit using a given dataset via jacobians.

ToDo: extract the repeated functions in a separate module for re-use.

"""

import os
import tensorflow as tf
import pickle
import argparse
from Quantum_model import *

import shutil
import datetime
from sklearn.model_selection import train_test_split
import math 
from scipy.special import logsumexp
import time

tf.config.run_functions_eagerly(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


import warnings
warnings.filterwarnings("ignore")


from quantumLayers import *


def parser_def():

    """_summary_

    Parser function to parse command line arguments

    Returns:
        argparse object: parser object.
    """
    
    parser = argparse.ArgumentParser(description='Parse parameters')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of Epochs')
    parser.add_argument('--num_qubits', type=int, default=10, help='Number of Epochs')
    parser.add_argument('--num_layers', type=int, default=10, help='Number of Epochs')
    parser.add_argument('--dataset', type=str, default='npy/10x10.pkl', help='Path to pkl file')
    parser.add_argument('--gradient_type', type=str, default='parameter-shift', help='Type of gradient [parameter-shift, finite-diff, adjoint]')
    parser.add_argument('--encoding_type', type=str, default='rx', help='Type of encoding [rx, rx_pi, rx_taninv]')
    parser.add_argument('--log_dir', type=str, default='check/logs', help='Path to TB logs')
    parser.add_argument('--arch', type=str, default='localized_encoding_multi_entang', help='Architecture model')
    parser.add_argument('--enc_layers', type=int, default=10, help='Number of Encoding layers split')
    parser.add_argument('--data_reuploading', type=str, default='No', help='To reupload Data')
    parser.add_argument('--shuffle', type=str, default='No', help='To shuffle values in image')
    parser.add_argument('--num_repeats', type=int, default=5, help='Number of Repeats')
    return parser


def load_dataset(args):

    """_summary_
    
    A function to load dataset from .npy file given as --dataset command line argument.

    Args:
        args (argparse.ArgumentParser): Parser object from parser_def function.

    Returns:
        train_dataset (tf.data.Dataset): data to be used for training as tensorflow dataset
        val_dataset (tf.data.Dataset): data to be used for validation as tensorflow dataset
        test_dataset (tf.data.Dataset): data to be used for testing as tensorflow dataset
        num_batches (int): number of batches in training set
        x_train.shape (list): shape of the train dataset.

    """

    with open(args.dataset, 'rb') as handle:
        x_train, y_train, x_test, y_test = pickle.load(handle).values()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    num_batches = math.ceil(len(x_train)/args.batch_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))


    SHUFFLE_BUFFER_SIZE = 1000

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(args.batch_size)
    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(args.batch_size)
    return train_dataset, val_dataset, test_dataset, num_batches, x_train.shape



def eff_dim(f_hat, n):
    """_summary_
    Get the effective dimension using the normalized fisher matrices

    Args:
        f_hat (np.array): fisher information matrix
        n (list): list of samples

    Returns:
        np.array: effective dimension
    """
    d = len(f_hat[0])
    effective_dim = []
    for ns in n:
        Fhat = f_hat * ns / (2 * np.pi * np.log(ns))
        one_plus_F = np.eye(d) + Fhat
        det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
        r = det / 2  # divide by 2 because of sqrt
        effective_dim.append(2 * (logsumexp(r) - np.log(100)) / np.log(ns / (2 * np.pi * np.log(ns))))
    return np.array(effective_dim)
    
    

if __name__ == '__main__':

    # Initialize parser
    parser = parser_def()
    args = parser.parse_args()
    args_dict = vars(args)
    
    # create log directory with times stamp if --log_dir argument is not passed
    if args.log_dir == 'logs/logs':
        args.log_dir +=  datetime.datetime.now().strftime("_%Y%m%d-%H:%M:%S")
    
    
    inputsize=100 # input size is hardcoded!!

    # set flags to repeat input data if data re-uploading technique is to be used.
    if args.data_reuploading == 'No':
        dRU = False
        args.data_reuploading = False
    else:
        dRU = True
        args.data_reuploading = True
        inputsize = 10

    # set flag to shuffle the features in each image a unique random order.
    if args.shuffle == 'No':
        dshuffle = False
    else:
        dshuffle = True
    
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)	


    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    #Load dataset
    train_dataset, val_dataset, test_dataset, num_batches, input_shape = load_dataset(args)
    
    # define the ecoding layer.
    encode_info_layer = encode_info(encoding_type=args.encoding_type)

    # load quantum circuit with initial parameters
    QcN = load_architecture(args)

    # Tensorflow interface to the quantum circuit.
    QC_layer = quantum_layer(QcN=QcN)
    
    tf_image_size = (10,10) # image dimension hard-coded !!!!

    model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=tf_image_size),
                encode_info_layer, 
                QC_layer])

    
    with open(args.dataset, 'rb') as handle:
        x_train, y_train, x_test, y_test = pickle.load(handle).values()

    
    @tf.function
    def train_step(x, y):
        """
        Function to calculate and return jacobian based on input x and label y
        """
        repeats = [10 for i in range(10)]
        with tf.GradientTape(persistent=True) as tape:
            logits = model(x, training=True)
            log_logits = tf.reshape(tf.math.log(logits), (10,))
            #input = tf.reshape(tf.repeat(logits, repeats), (10,10))
            #loss_value = loss_fn(y[0][1], logits)
        """jac_list = []
        for i in range(len(input)):
            intt_c = input[i]
            jac = tape.jacobian(logits, model.trainable_weights)
            jac = jac*tf.math.sqrt(logits)
            jac_list.append(jac)"""
            
        jac = tape.jacobian(log_logits, model.trainable_weights)
        jac = tf.squeeze(jac)
        jac = tf.reshape(jac, (jac.shape[0], np.prod(tf.shape(jac).numpy()[1:])))
        #t_logits = tf.math.sqrt(tf.tile(tf.transpose(logits), (1,jac.shape[1])))
        res =  tf.Variable(tf.zeros(jac.shape))
        logits_num = logits.numpy()[0]
        for i in range(len(logits_num)):
            res[i].assign(jac[i]*np.sqrt(logits_num[i]))
        
        return res

    # These values are hard coded. Should implement a better option.
    grads_list = []
    num_inputs = 400
    num_theatas = 200
    if dRU:
        input_size = 10
    else:
        input_size = 100
    output_size = 10
    num_parameters = 200


    tf_image_size = (10,10)
    x = np.zeros((num_inputs,)+tf_image_size)
    x_label = np.zeros((num_inputs))
    
    i=0
    num_images = 0
    remaind = num_inputs%10

    # repeat the inputs based on required number of inputs and number of parameters.
    if num_inputs>=10:
        num_images = int(num_inputs/10)
        for i in range(10):
            label = np.zeros(10)
            label[i] = 1
            ii = np.where(y_train == label)[0]
            img = x_train[ii[0:num_images]] #.reshape(num_images,input_size)
            if input_size != 100:
                img = np.tile(img, 10).reshape(img.shape[0], 10,10)
            x[i*num_images:i*num_images+num_images,:] = img
            x_label[i*num_images:i*num_images+num_images] = i
        #x = np.random.normal(0, 1, size=(num_inputs, self.model.inputsize))
    if remaind != 0:
        x[i*num_images:i*num_images+remaind,:] = x_train[:remaind].reshape(remaind,100)
        x_label[i*num_images:i*num_images+remaind] = [np.where(y == 1.0)[0][0] for y in y_train[-remaind:]]
    
    x = np.tile(x, (num_theatas, 1, 1))
    
    gradients = np.zeros((len(x), output_size, num_parameters))
    counter = 0
    eff_counter = 0
    with open(args.log_dir+'eff_dim.txt', 'w') as f:
        f.write("Eff dim and trace \n")
    

    # calculate the fisher information matrix for all images.
    for epoch in range(len(x)):
        #for theta in range(num_theatas):
        start = time.time()
        encode_info_layer = encode_info(encoding_type=args.encoding_type)
        QcN = load_architecture(args)
        QC_layer = quantum_layer(QcN=QcN)
        model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=tf_image_size),
                    encode_info_layer, 
                    QC_layer, tf.keras.layers.Softmax()])

        """model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=tf_image_size),
                    tf.keras.layers.Dense(10, use_bias=False) 
                    , tf.keras.layers.Softmax()])"""
        
        counter = counter + 1
        x_t = tf.constant(x[epoch,:].reshape(1,100))
        y_t = tf.constant(1)
        grads = train_step(x_t, y_t)
        gradients[epoch,:,:] = grads.numpy()
        end = time.time()
        print(counter, end-start) 
    

        if (epoch+1)%(num_theatas*20) == 0 and epoch>0:
            eff_counter += 1
            fishers = np.zeros((epoch+1, num_parameters, num_parameters))
            for i in range(epoch+1):
                grads = gradients[i]
                temp_sum = np.zeros((output_size, num_parameters, num_parameters))
                for j in range(output_size):
                    temp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
                fishers[i] += np.sum(temp_sum, axis=0)

            trace = np.trace(np.average(fishers, axis=0))  # compute the trace with all fishers
            
            # average the fishers over the num_inputs to get the empirical fishers
            fisher = np.average(np.reshape(fishers, (num_theatas, eff_counter*20, num_parameters, num_parameters)), axis=1)
            
            f = num_parameters * fisher / trace  # calculate f_hats for all the empirical fishers
            
            print("trace :", trace)

            n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]

            eff_d = eff_dim(f_hat=f, n=n)/num_parameters
            print("eff_dm :",  eff_d)
            with open(args.log_dir+'eff_dim.txt', 'a') as output_file:
                output_file.write("Epoch : {} ".format(epoch))
                output_file.write("Trace : {} ".format(trace))
                output_file.write("Eff_dim : ")
                for eff in eff_d:
                    output_file.write(str(eff) + ' ')
                output_file.write("\n")
            result_dict = {
                'trace ': trace,
                'fisher': f,
                'effdim':eff_d,
                'grad': gradients
            }
            # Save intermediate results with less number of samples.
            with open(args.log_dir+'/result_'+str(epoch+1)+'.pkl', 'wb') as handle:
                pickle.dump( result_dict
                    , handle, protocol=pickle.HIGHEST_PROTOCOL)
        if epoch == 3999:
            break
        
    gradients = np.array(gradients)
    gradients = gradients.reshape((num_inputs*num_theatas,output_size, np.prod(gradients.shape[2:])))

    fishers = np.zeros((len(gradients), num_parameters, num_parameters))
    for i in range(len(gradients)):
        grads = gradients[i]
        temp_sum = np.zeros((output_size, num_parameters, num_parameters))
        for j in range(output_size):
            temp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
        fishers[i] += np.sum(temp_sum, axis=0)

    trace = np.trace(np.average(fishers, axis=0))  # compute the trace with all fishers
    # average the fishers over the num_inputs to get the empirical fishers
    fisher = np.average(np.reshape(fishers, (num_theatas, num_inputs, num_parameters, num_parameters)), axis=1)
    f = num_parameters * fisher / trace  # calculate f_hats for all the empirical fishers
    
    print(trace)

    n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]

    eff_d = eff_dim(f_hat=f, n=n)/num_parameters
    print(eff_d)
    
    
    result_dict = {
        'trace ': trace,
        'fisher': f,
        'effdim':eff_d,
        'grad': gradients
    }
    with open(args.log_dir+'/result.pkl', 'wb') as handle:
        pickle.dump( result_dict
            , handle, protocol=pickle.HIGHEST_PROTOCOL)