################################################
################################################
####                                        #### 
####     KERAS IMPLEMENTATION OF METRICS    ####
####                                        ####
################################################
################################################


# Python modules
from keras import backend as K
import tensorflow as tf
import numpy as np
from itertools import product
from functools import partial, update_wrapper


        

###########################################################
###########################################################
####                                                   #### 
####                WEIGHTED CAT CROSSENTROPY          ####
####                by Andreas Maunz                   ####
#### PIO-224                                           ####
#### https://github.com/keras-team/keras/issues/2115   ####
####                                                   ####
###########################################################
###########################################################


def _ones_square_np(n):
  return np.ones((n,n), dtype=np.float32)



# Calculates weighted categorical crossentropy based on float-32
# Each pair of true and predicted values are assigned a weight based on
# categorical crossentropy and multiplied by a user-defined weight.
# Weights are encoded as nxn tensor, where i-j entry indicates
# the weight for predicting j when the actual value is i.
#
# @param y_true: true values in one-hot encoding, ex: [[1.,0.], [0.,1.]]
# @param y_pred: predicted values of the same shape as y_true
# @param weights: square matrix of size number-of-classes, e.g. tf.ones(2)
# @return weighted and unweighted categorical crossentropy (length(y_true)) at positions 1, 2

def _w_categorical_crossentropy(y_true, y_pred, weights):
    number_classes = weights.get_shape()[0] # weights is square matrix

    # calculate zeros with length equal the number of preds
    # y_pred[:, 0] = 1st position of rows, i.e. preds for 1st class
    final_mask = K.zeros_like(y_pred[:, 0]) 

    # get max prediction value for each case
    y_pred_max = K.max(y_pred, axis=1)           
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1)) # Transpose

    # get maximum location (nr_cases x nr_classes, binary)
    y_pred_max_mat = K.equal(y_pred, y_pred_max) 
    y_pred_max_mat = tf.cast(y_pred_max_mat, tf.float32)
    
    # assign weights to preds, depending on being true or false
    for c_p, c_t in product(range(number_classes), range(number_classes)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])

    # weight the cce
    cce = K.categorical_crossentropy(y_pred, y_true)
    wcce = cce * final_mask
    return wcce, cce, final_mask



# Utility to create a partial function
# Wrapper for actual code, returns only the weighted cce value
# see transfer_learning.py on how it is used

def _wcce(y_true, y_pred, weights):
    return _w_categorical_crossentropy(y_true, y_pred, weights)[0]

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func    

