import tensorflow as tf
import os
from PIL import Image
import numpy as np
import pickle




"""mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train, x_test
name = 'mnist'"""

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train, x_test
name = 'fashion_mnist'


data_location = 'npy/'
if not os.path.exists(data_location):
    os.makedirs(data_location)



scale = False
average = True
resize = True

onehot_labels = True

scale_factor = 1.0/10

if not scale:
    scale_factor = 1.0

if resize:
    final_img_w = 10
    final_img_h = 10
else:
    final_img_w = x_train.shape[1]
    final_img_h = x_train.shape[2]
num_clases = np.unique(y_train).size



final_img_train = np.zeros((len(x_train),final_img_w, final_img_h))
final_img_test = np.zeros((len(x_test),final_img_w, final_img_h))

for i in range(len(x_train)):
    if resize:
        im = Image.fromarray(x_train[i,:,:], mode='L')
        im = im.resize((final_img_w,final_img_h), resample=Image.LANCZOS)
        final_img_train[i] = np.asarray(im)*(1./255)*scale_factor
    else:
        final_img_train[i] = np.asarray(x_train[i,:,:])*(1./255)*scale_factor


for i in range(len(x_test)):
    if  resize:
        im = Image.fromarray(x_test[i,:,:], mode='L')
        im = im.resize((final_img_w,final_img_h), resample=Image.LANCZOS)
        final_img_test[i] = np.asarray(im)*(1./255)*scale_factor
    else:
        final_img_test[i] = np.asarray(x_test[i,:,:])*(1./255)*scale_factor

if average:
    final_img_train = np.sum(final_img_train, axis=1)
    final_img_test = np.sum(final_img_test, axis=1)

if onehot_labels:
    labels_train = np.zeros((y_train.size, num_clases))
    labels_train[np.arange(y_train.size),y_train] = 1

    labels_test = np.zeros((y_test.size,num_clases))
    labels_test[np.arange(y_test.size),y_test] = 1
else:
    labels_train = y_train
    labels_test = y_test

if scale:
    pkl_file = 'npy/'+name+'_'+str(final_img_w)+'x'+str(final_img_h)+'_scaled.pkl'
else:
    pkl_file = 'npy/'+name+'_'+str(final_img_w)+'x'+str(final_img_h)+'.pkl'


if average:
    pkl_file = pkl_file[:-4]+'_colMean.pkl'

with open(pkl_file, 'wb') as handle:
    pickle.dump(
        {'x_train': final_img_train,
         'y_train': labels_train,
         'x_test': final_img_test,
         'y_test': labels_test}
        , handle, protocol=pickle.HIGHEST_PROTOCOL) 
