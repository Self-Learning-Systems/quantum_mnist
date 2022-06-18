"""
Author: Maniraman Periyasamy
Organization: Maniraman Periyasamy

This file implements the tensroflow quantum layer from the quantum circuit implemented using Quantum_model.py file.

"""

import tensorflow as tf
import numpy as np
import tensorflow_quantum as tfq
import cirq



class quantum_layer(tf.keras.Model):
  
    def __init__(self, QcN, quantum_weight_initialization = 'random', name='quantum_layer'):
        """_summary_

        Args:
            QcN (QcModel object): Quantum circuit model object from Quantum_model.py file which includes the variational circuit
            quantum_weight_initialization (str, optional): Weight initialization method. Defaults to 'random'.
            name (str, optional): name of the layer. Defaults to 'quantum_layer'.
        """
        super().__init__(name=name)


        self.QcN = QcN
        self.QcN.build_module(add_encoding=True)
        
        # initialize weights
        if quantum_weight_initialization == 'random': self.trainable_parameters = np.random.uniform(low=0.0, high=1, size=(1,len(self.QcN.weight_parameters))).astype(np.float32)
        elif quantum_weight_initialization == 'zeros': self.trainable_parameters = np.zeros((1,len(self.QcN.weight_parameters))).astype(np.float32)
        elif quantum_weight_initialization == 'ones': self.trainable_parameters = np.ones((1,len(self.QcN.weight_parameters))).astype(np.float32)

        # Map parameters to the weights
        self.trainable_parameters = tf.Variable(initial_value=self.trainable_parameters, trainable=True, name="weights", dtype=tf.dtypes.float32)
        symbols = [str(symb) for symb in self.QcN.weight_parameters + self.QcN.encoding_parameters]
        #symbols = [str(symb) for symb in self.QcN.weight_parameters]
        
        self.indices = tf.constant([sorted(symbols).index(a) for a in symbols])
        # construct empty sircuit place holder
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

    
    def call(self, inputs):
        # map and tile input parameters to its gates.

        batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_weights = tf.tile(self.trainable_parameters, multiples=[batch_dim, 1])
        joined_vars = tf.concat([tiled_up_weights, inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        res = self.QcN.QC([tiled_up_circuits, joined_vars])
        return res

class data_reuploading(tf.keras.layers.Layer):

    def __init__(self, num_layers = 4, name='data_reuploading_layer'):
        """_summary_
        Layer to tile the input in case of data re-uploading
        Args:
            num_layers (int, optional): number of layers. Defaults to 4.
            name (str, optional): name of layer. Defaults to 'data_reuploading_layer'.
        """
        super().__init__(name=name)

        self.num_layers = num_layers

    def call(self, x):

        x =  tf.tile(x, multiples=[1, self.num_layers])
        """if tf.rank(x) == 2:
            x = tf.expand_dims(x, axis=0)"""
        return x

class shuffle(tf.keras.layers.Layer):

    def __init__(self, batch_size=32, input_size=100, seed = 42, name='shuffle_layer'):
        """_summary_

        Layer to shuffle the features of an image

        Args:
            batch_size (int, optional): number of images in a batch. Defaults to 32.
            input_size (int, optional): input image size (total number of pixels). Defaults to 100.
            seed (int, optional): random seed to be used for selecting the shuffling order. Defaults to 42.
            name (str, optional): name of the layer. Defaults to 'shuffle_layer'.
        """
        super().__init__(name=name)

        self.seed = seed
        self.batch_size = batch_size
        self.input_size = input_size
        np.random.seed(seed)
        ind = np.arange(input_size)
        np.random.shuffle(ind)
        indices = []
        for i in range(self.batch_size):
            for j in range(input_size):
                indices.append([i, ind[j]])
        self.indices = indices
    
    
    def call(self, x):
        #perm = [1,0]
        #value = tf.random.shuffle(tf.transpose(x, perm=perm), seed = self.seed)
        #value = tf.transpose(value, perm=perm)
        value = tf.reshape(tf.gather_nd(x, indices=self.indices), [self.batch_size, self.input_size])
        return value


class padding_1D(tf.keras.layers.Layer):

    def __init__(self, output_dim = 30, name='padding_1D_layer'):
        """_summary_
        Padding layer to be used
        Args:
            output_dim (int, optional): required output dimension. Defaults to 30.
            name (str, optional): name of the layer. Defaults to 'padding_1D_layer'.
        """
        super().__init__(name=name)
        
        self.output_dim = output_dim

    def call(self, x):
        shape = x.get_shape().as_list()
        if shape[1] < self.output_dim:
            zeros_to_add = self.output_dim - shape[1]
            paddings = tf.constant([[0, 0,], [0, zeros_to_add]])
            x = tf.pad(x, paddings=paddings)
            #x = tf.concat([x, tf.zeros(zeros_to_add)])
        return x

        
class encode_info(tf.keras.Model):

    def __init__(self,encoding_type, name='encode_info_layer'):
        """_summary_
        This layer defines the type of encoding to be used
        Args:
            encoding_type (string): Type of encoding.
            name (str, optional): name of the layer. Defaults to 'encode_info_layer'.
        """
        super().__init__(name=name)
        
        self.encoding_type = encoding_type
        
    def call(self, inputs):
        if self.encoding_type == 'rx':
            res = inputs
        elif self.encoding_type == 'rx_pi':
            res = tf.math.scalar_mul(scalar=np.pi, x=inputs, name='scalar_multiply_layer')
        elif self.encoding_type == 'rx_taninv':
            res = tf.math.atan(x=inputs, name='ArcTan layer')
        return res





