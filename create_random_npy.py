from typing import final
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import pickle
from sklearn.datasets import make_classification


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

(x_train, y_train) = make_classification(n_samples=5000, n_features=22, n_informative=16, n_classes=5, class_sep=1.5)
(x_test, y_test) = make_classification(n_samples=5000, n_features=22, n_informative=16, n_classes=5 , class_sep=1.5)

data_location = 'npy/'
if not os.path.exists(data_location):
    os.makedirs(data_location)

x_train = NormalizeData(x_train)
x_test = NormalizeData(x_test)


scale = False
average = False

onehot_labels = True

scale_factor = 1.0/10

if not scale:
    scale_factor = 1.0

num_clases = np.unique(y_train).size

final_img_w = 1
final_img_h = x_train.shape[1]

if onehot_labels:
    labels_train = np.zeros((y_train.size, num_clases))
    labels_train[np.arange(y_train.size),y_train] = 1

    labels_test = np.zeros((y_test.size,num_clases))
    labels_test[np.arange(y_test.size),y_test] = 1
else:
    labels_train = y_train
    labels_test = y_test

if scale:
    pkl_file = 'npy/'+str(final_img_w)+'x'+str(final_img_h)+'_scaled.pkl'
else:
    pkl_file = 'npy/'+str(final_img_w)+'x'+str(final_img_h)+'.pkl'


if average:
    pkl_file = pkl_file[:-4]+'_colMean.pkl'

with open(pkl_file, 'wb') as handle:
    pickle.dump(
        {'x_train': x_train,
         'y_train': labels_train,
         'x_test': x_test,
         'y_test': labels_test}
        , handle, protocol=pickle.HIGHEST_PROTOCOL) 
