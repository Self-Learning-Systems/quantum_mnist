"""

Author: Maniraman Periyasamy
Organization: Maniraman Periyasamy

This file implements a simple supervised learning model with different quantum circuits as function approximation.

"""

import tensorflow as tf
import pickle
import argparse
from Quantum_model import *
import os
import shutil
import datetime
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Progbar
import math
import pandas as pd

tf.config.run_functions_eagerly(True)

from quantumLayers import *


def parser_def():
    """_summary_

    Parser function to parse command line arguments

    Returns:
        argparse object: parser object.
    """
    parser = argparse.ArgumentParser(description='Parse parameters')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of Epochs')
    parser.add_argument('--num_qubits', type=int, default=10, help='Number of Epochs')
    parser.add_argument('--num_layers', type=int, default=10, help='Number of Epochs')
    parser.add_argument('--dataset', type=str, default='npy/10x10.pkl', help='Path to pkl file')
    parser.add_argument('--gradient_type', type=str, default='parameter-shift', help='Type of gradient [parameter-shift, finite-diff, adjoint]')
    parser.add_argument('--encoding_type', type=str, default='rx_pi', help='Type of encoding [rx, rx_pi, rx_taninv]')
    parser.add_argument('--log_dir', type=str, default='check/logs', help='Path to TB logs')
    parser.add_argument('--arch', type=str, default='localized_encoding_multi', help='Architecture model')
    parser.add_argument('--enc_layers', type=int, default=1, help='Number of Encoding layers split')
    parser.add_argument('--data_reuploading', type=str, default='No', help='To reupload Data')
    parser.add_argument('--shuffle', type=str, default='No', help='To shuffle values in image')
    parser.add_argument('--num_repeats', type=int, default=3, help='Number of Repeats')
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
    test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(args.batch_size)
    return train_dataset, val_dataset, test_dataset, num_batches, x_train.shape


if __name__ == '__main__':

    # Load parser
    parser = parser_def()
    args = parser.parse_args()
    args_dict = vars(args)
    
    # create log directory with times stamp if --log_dir argument is not passed
    if args.log_dir == 'logs/logs':
        args.log_dir +=  datetime.datetime.now().strftime("_%Y%m%d-%H:%M:%S")
    
    # set flags to repeat input data if data re-uploading technique is to be used.
    if args.data_reuploading == 'No':
        dRU = False
        args.data_reuploading = False
    else:
        dRU = True
        args.data_reuploading = True
    
    # set flag to shuffle the features in each image a unique random order.
    if args.shuffle == 'No':
        dshuffle = False
    else:
        dshuffle = True
    
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)	

    # create model checkpoints directory
    checkpoint_dir = args.log_dir+'/model_checkpoints'

    # arrays to accumulate training and validation results
    train_acc_repeats = np.zeros((args.num_repeats, args.num_epochs))
    val_acc_repeats = np.zeros((args.num_repeats, args.num_epochs))
    train_loss_repeats = np.zeros((args.num_repeats, args.num_epochs))
    val_loss_repeats = np.zeros((args.num_repeats, args.num_epochs))
    
    w = tf.summary.create_file_writer(args.log_dir)
    eval_loss = []
    eval_acc = []


    # Repeat training multiple times and average repeatability.
    for repeat in range(args.num_repeats):
        
        #Load dataset
        train_dataset, val_dataset, test_dataset, num_batches, input_shape = load_dataset(args)
        
        # define the ecoding layer.
        encode_info_layer = encode_info(encoding_type=args.encoding_type)
        # load quantum circuit with initial parameters
        QcN = load_architecture(args)

        # Tensorflow interface to the quantum circuit.
        QC_layer = quantum_layer(QcN=QcN)

        # initialize padding layer. used if needed
        padd_layer = padding_1D(output_dim=args.num_layers*args.num_qubits)

        # initialize shuffling layer to shuffle features of the input image. used if requested.
        shuffle_layer = shuffle(batch_size=args.batch_size, input_size=args.num_layers*args.num_qubits, seed=42)
        
        
        # set the combination of layers (data re-uploading, shuffle, pad etc.) and construct quantum based on parser arguments.
        if dRU:
            tf_image_size = input_shape[1:]
            data_reuploading_layer = data_reuploading(args.num_layers)
            if dshuffle:
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=tf_image_size), padd_layer, shuffle_layer, data_reuploading_layer,
                    encode_info_layer, 
                    QC_layer])
            else:
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=tf_image_size), padd_layer, data_reuploading_layer,
                    encode_info_layer, 
                    QC_layer])
        else:
            tf_image_size = input_shape[1:]
            if dshuffle:
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=tf_image_size), padd_layer, shuffle_layer,
                    encode_info_layer, 
                    QC_layer])
            else:
                model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=tf_image_size), padd_layer,
                    encode_info_layer, 
                    QC_layer])
        
        # loss function
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam',loss=loss_fn, metrics=['accuracy'])
        

        # model checkpoints to use
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir,
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True)
        earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6)

        # train and accumulate results
        hist = model.fit(train_dataset, validation_data=val_dataset, epochs=args.num_epochs, callbacks=[model_checkpoint_callback])
        train_history = hist.history
        train_loss_repeats[repeat, 0:len(train_history['loss'])] = train_history['loss']
        train_acc_repeats[repeat, 0:len(train_history['loss'])] = train_history['accuracy']
        val_loss_repeats[repeat, 0:len(train_history['loss'])] = train_history['val_loss']
        val_acc_repeats[repeat, 0:len(train_history['loss'])] = train_history['val_accuracy']
        
        
        hist_eval = model.evaluate(test_dataset, verbose=1, batch_size=args.batch_size, return_dict=True)
        eval_loss.append(hist_eval['loss'])
        eval_acc.append(hist_eval['accuracy'])

        # store the results as csv
        df_repeat = pd.DataFrame.from_dict(train_history)
        df_repeat.to_csv(args.log_dir+'/metrics_'+str(repeat)+'.csv', index=False)

    # store averaged results and plot
    overall_train_loss = np.mean(train_loss_repeats, axis=0)
    overall_train_acc = np.mean(train_acc_repeats, axis=0)
    overall_val_loss = np.mean(val_loss_repeats, axis=0)
    overall_val_acc = np.mean(val_acc_repeats, axis=0)

    args_dict['eval_loss'] = eval_loss
    args_dict['eval_acc'] = eval_acc
    args_dict['eval_loss_mean'] = np.mean(eval_loss)
    args_dict['eval_acc_mean'] = np.mean(eval_acc)

    # uncomment if tensorboard is to be used.
    """with w.as_default():
        for i in range(args.num_epochs):
            tf.summary.scalar('train_loss',overall_train_loss[i],i)
            tf.summary.scalar('train_acc',overall_train_acc[i],i)
            tf.summary.scalar('val_loss',overall_val_loss[i],i)
            tf.summary.scalar('val_acc',overall_val_acc[i],i)"""
    result_dict={
        "epochs": np.array(np.arange(len(overall_train_acc))),
        "loss":overall_train_loss,
        "accuracy":overall_train_acc,
        "val_loss":overall_val_loss,
        "val_acc":overall_val_acc
    }

    df = pd.DataFrame.from_dict(result_dict)
    df.to_csv(args.log_dir+'/metrics.csv', index=False)
    
    # save the parameters and arguments used to construct this model for future reference and repeatability.
    with open(args.log_dir+'/experiment_args.json', 'w') as fp:
        json.dump(args_dict, fp, sort_keys=True, indent=4, separators=(',', ': '))

    
