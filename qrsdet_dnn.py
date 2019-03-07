#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:39:24 2019

@author: mariano
"""
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.layers import Flatten, Conv1D, GlobalMaxPooling1D, MaxPooling1D

from keras.callbacks import Callback
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import backend as K
from keras.optimizers import Adam
#from keras.utils import Sequence, multi_gpu_model
 
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import os
from glob import glob
import time


def generator_class( datasets, batch_size=32):
    """A generator yields (source, target) arrays for training."""

    while True:
          
        cant_ds = len(datasets)
    
        # Shuffle datasets
        datasets = np.random.choice(datasets, cant_ds, replace=False )
        
        for ds_idx in range(cant_ds):
            
            this_ds = datasets[ds_idx]
#            print('\nEntering:' + this_ds + '\n')
            train_ds = np.load(this_ds)[()]
            train_x = train_ds['signals']
            cant_samples = train_x.shape[0]
            train_x = train_x.reshape(cant_samples, 1, train_x.shape[1])
            train_y = train_ds['labels']
            train_y = train_y.flatten()
    
            samples_idx = np.random.choice(np.arange(cant_samples), cant_samples, replace=False )
        
            for ii in range(0, cant_samples, batch_size):
                # Get the samples you'll use in this batch
                xx = np.array(train_x[ samples_idx[ii:ii+batch_size],:,:], dtype='double') 
                yy = np.array(train_y[ samples_idx[ii:ii+batch_size] ], dtype='double') 
          
                yield ( xx, yy )


class MyCallbackClass(Callback):
    
    def on_train_begin(self, logs={}):
     self.val_f1s = []
     self.val_recalls = []
     self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
     val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
#     val_predict = (np.asarray(self.model.p (self.validation_data[0]))).round()

     val_targ = self.validation_data[1]
     _val_f1 = f1_score(val_targ, val_predict)
     _val_recall = recall_score(val_targ, val_predict)
     _val_precision = precision_score(val_targ, val_predict)
     
     self.val_f1s.append(_val_f1)
     self.val_recalls.append(_val_recall)
     self.val_precisions.append(_val_precision)
     print('\nval_f1: {:3.3f} — val_precision: {:3.3f} — val_recall: {:3.3f}'.format(  _val_f1, _val_precision, _val_recall ) )
           
     return

def se(y_true, y_pred):
    """Recall or sensitivity metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def pp(y_true, y_pred):
    """Precision or Positive Predictive Value metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    

    precision = pp(y_true, y_pred)
    recall = se(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_dataset_size(train_list_fn):
    
    if os.path.isfile( train_list_fn ):
        try:
            paths = np.loadtxt(train_list_fn, dtype=str )
        except:
            paths = glob(train_list_fn)
    else:
        paths = glob(train_list_fn)
    
    if paths == []:
        raise EnvironmentError
    else:
        cant_train_parts = len(paths)
        train_samples = 0
        for ii in range(cant_train_parts):
            train_ds = np.load( paths[ii] )[()]
            train_samples += train_ds['cant_total_samples']
        
        win_size_samples = (train_ds['signals']).shape[1]
        
    return train_samples, win_size_samples, paths

parser = ap.ArgumentParser(description='Prueba para entrenar un detector de QRS mediante técnicas de deep learning')

parser.add_argument( '--train_list', 
                     default='', 
                     type=str, 
                     help='Nombre de la base de datos')

parser.add_argument( '--val_list', 
                     default='', 
                     type=str, 
                     help='Nombre de la base de datos')

parser.add_argument( '--test_list', 
                     default='', 
                     type=str, 
                     help='Nombre de la base de datos')

args = parser.parse_args()

train_list_fn = args.train_list
val_list_fn = args.val_list
test_list_fn = args.test_list



cant_filtros = 12
size_filtros = 3
hidden_dims  = 6
batch_size = 2**8
epochs = 50


## Train

if train_list_fn == '':
    raise EnvironmentError

train_samples, train_features, train_paths = get_dataset_size(train_list_fn)

train_generator = generator_class(train_paths, batch_size)

if val_list_fn == '':
    
    val_generator = []
    
else:
    
    val_samples, val_features, val_paths = get_dataset_size(val_list_fn)
    
    val_generator = generator_class(val_paths, batch_size)
    
if test_list_fn != '':
    
    test_generator = []
    
else:
    
    val_samples, val_features, val_paths = get_dataset_size(val_list_fn)
    
    test_generator = generator_class(val_paths, batch_size)
    


#    ## Validation
#    paths = glob(os.path.join(ds_config['dataset_path'], 'ds_val_part_*.npy' ))
#
#    if paths == []:
#        raise EnvironmentError
#    else:
#        
#        cant_val_parts = len(paths)
#        train_ds = np.load(os.path.join(ds_config['dataset_path'], 'ds_val_part_' + str(cant_val_parts) + '.npy' ))[()]
#        val_samples = train_ds['cant_total_samples']
#        
#        val_generator = generator_class(paths, batch_size);
#
#    ## Test
#    
#    paths = glob(os.path.join(ds_config['dataset_path'], 'ds_test_part_*.npy' ))
#
#    if paths == []:
#        raise EnvironmentError
#    else:
#        
#        cant_test_parts = len(paths)
#        train_ds = np.load(os.path.join(ds_config['dataset_path'], 'ds_test_part_' + str(cant_test_parts) + '.npy' ))[()]
#        test_samples = train_ds['cant_total_samples']
#        
#        test_generator = generator_class(paths, batch_size);


## Debug signals in train and val sets
#plt.figure(1); idx = np.random.choice(np.array((train_y==1).nonzero()).flatten(), 20, replace=False ); sigs = train_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))
#plt.figure(1); idx = np.random.choice(np.array((train_y==0).nonzero()).flatten(), 20, replace=False ); sigs = train_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))
#
#plt.figure(1); idx = np.random.choice(np.array((val_y==1).nonzero()).flatten(), 100, replace=False ); sigs = val_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))
#plt.figure(1); idx = np.random.choice(np.array((val_y==0).nonzero()).flatten(), 100, replace=False ); sigs = val_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))    


print('Build model...')

with tf.device('/cpu:0'):
    model = Sequential()

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(cant_filtros,
                     size_filtros,
                     input_shape=(1, train_features),
                     strides=1,
                     padding='same',
                     activation='relu'
                     ))
    
#    model.add(Conv1D(cant_filtros,
#                     size_filtros,
#                     padding='same',
#                     activation='relu'
#                     ))
    
#    model.add(Dropout(0.25))
    
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    
    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
#    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    

# Replicates the model on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
#    parallel_model = multi_gpu_model(model, gpus=2)
#    parallel_model.compile(loss='binary_crossentropy',
#                              optimizer=Adam(lr=0.001),
#                              metrics=[f1, pp, se])

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=[f1, pp, se])

#my_callback = MyCallbackClass()

print('Start training @ ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))
start_time = time.time()

history = model.fit_generator(train_generator,
                              steps_per_epoch = np.ceil(train_samples / batch_size),
                              epochs = epochs,
                              validation_data = val_generator,
                              validation_steps = 500
#                              callbacks=[my_callback]
                              )

#    history = parallel_model.fit_generator(train_generator,
#                                  steps_per_epoch = np.ceil(train_samples / batch_size),
#                                  epochs = 50
#        #                          validation_data=(train_x, train_y),
##                                  callbacks=[my_callback])
#                                  )

#    history = parallel_model.fit_generator(train_generator,
#                                  steps_per_epoch = np.ceil(train_samples / batch_size),
#                                  validation_data=val_generator,
#                                  validation_steps = np.ceil(val_samples / batch_size),
#                                  epochs = 10
#        #                          validation_data=(train_x, train_y),
##                                  callbacks=[my_callback])
#                                  )
#    


print('End training @ ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))
time_elapsed = time.time() - start_time
print( 'Time elapsed to train: ' + time.strftime("%H:%M:%S", time.gmtime(time_elapsed)) )

result_path = os.path.join('.', 'results')
os.makedirs(result_path, exist_ok=True)

model_id = time.strftime("%d_%b_%Y_%H_%M_%S", time.gmtime())
model.save( os.path.join( result_path, model_id + 'qrs_detector_model_' + '.h5'))  # creates a HDF5 file 'my_model.h5'
np.save( os.path.join( result_path, model_id + '_history.npy'), {'history' : history})

train_se = history.history['se']
#val_se = my_callback.val_recalls

train_pp = history.history['pp']
#val_pp = my_callback.val_precisions

train_f1 = history.history['f1']
#val_f1 = my_callback.val_f1s

train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, epochs + 1)

# Visualize accuracy history
    
# Visualize accuracy history
#plt.figure(1)
#plt.plot(epoch_count, np.transpose(np.array((train_f1, train_se, train_pp, val_f1, val_se, val_pp))))
#plt.legend(['train_f1', 'train_se', 'train_pp', 'val_f1', 'val_se', 'val_pp'])
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy Score')
#plt.title('F1');
#plt.show();

plt.figure(2)
plt.plot(epoch_count, train_loss, 'r--')
plt.plot(epoch_count, val_loss, 'b-')
plt.legend(['Train', 'Val'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss');
plt.show();


train_predict = (np.asarray(model.predict_generator(train_generator, steps_per_epoch = np.ceil(train_samples / batch_size) ))).round()
train_f1 = f1_score(train_y, train_predict)
train_recall = recall_score(train_y, train_predict)
train_precision = precision_score(train_y, train_predict)
print('Train\n-----\n F1: {:3.3f} — +P: {:3.3f} — Se: {:3.3f}'.format(  train_f1, train_precision, train_recall ) )

val_predict = (np.asarray(model.predict(val_x))).round()
val_f1 = f1_score(val_y, val_predict)
val_recall = recall_score(val_y, val_predict)
val_precision = precision_score(val_y, val_predict)
print('Validation\n----------\n F1: {:3.3f} — +P: {:3.3f} — Se: {:3.3f}'.format( val_f1, val_precision, val_recall ) )

test_ds = np.load(test_filename)[()]
test_recs = test_ds['recordings']
test_x = test_ds['signals']
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
test_y = test_ds['labels']
test_y = test_y.flatten()

test_predict = (np.asarray(model.predict(test_x))).round()
test_f1 = f1_score(test_y, test_predict)
test_recall = recall_score(test_y, test_predict)
test_precision = precision_score(test_y, test_predict)
print('Test\n----\n F1: {:3.3f} — +P: {:3.3f} — Se: {:3.3f}'.format(  test_f1, test_precision, test_recall ) )

plt.figure(3)
plt.plot(range(3), np.transpose(np.array(((train_f1, val_f1, test_f1), (train_recall, val_recall, test_recall), (train_precision, val_precision, test_precision) ))), 'o--' )
plt.xticks(np.arange(3), ('Train', 'Val', 'Test'))
plt.legend(['F1', 'Se', '+P'])
plt.title('Performance en los datasets');
plt.show();

