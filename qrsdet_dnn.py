#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:39:24 2019

@author: mariano
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv1D, GlobalMaxPooling1D, MaxPooling1D, BatchNormalization

#from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, TerminateOnNaN, LearningRateScheduler
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
from pandas import DataFrame, read_csv


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
            train_x = train_x.reshape(cant_samples, train_x.shape[1], 1)
            train_y = train_ds['labels']
            train_y = train_y.flatten()
    
            samples_idx = np.random.choice(np.arange(cant_samples), cant_samples, replace=False )
        
            for ii in range(0, cant_samples, batch_size):
                # Get the samples you'll use in this batch
                xx = np.array(train_x[ samples_idx[ii:ii+batch_size],:,:], dtype='double') 
                yy = np.array(train_y[ samples_idx[ii:ii+batch_size] ], dtype='double') 
          
                yield ( xx, yy )

            

#class MyCallbackClass(Callback):
#    
#    def on_train_begin(self, logs={}):
#     self.val_f1s = []
#     self.val_recalls = []
#     self.val_precisions = []
#     
#    def on_epoch_end(self, epoch, logs={}):
#     val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
##     val_predict = (np.asarray(self.model.p (self.validation_data[0]))).round()
#
#     val_targ = self.validation_data[1]
#     _val_f1 = f1_score(val_targ, val_predict)
#     _val_recall = recall_score(val_targ, val_predict)
#     _val_precision = precision_score(val_targ, val_predict)
#     
#     self.val_f1s.append(_val_f1)
#     self.val_recalls.append(_val_recall)
#     self.val_precisions.append(_val_precision)
#     print('\nval_f1: {:3.3g} — val_precision: {:3.3g} — val_recall: {:3.3g}'.format(  _val_f1, _val_precision, _val_recall ) )
#           
#     return

def t_se(y_true, y_pred):
    """Recall or sensitivity metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def t_pp(y_true, y_pred):
    """Precision or Positive Predictive Value metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def t_f1(y_true, y_pred):
    

    precision = t_pp(y_true, y_pred)
    recall = t_se(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def pp(y_true, y_pred):
    """Precision or Positive Predictive Value metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + np.finfo(predicted_positives.dtype).resolution )
    return precision

def se(y_true, y_pred):
    """Recall or sensitivity metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + np.finfo(possible_positives.dtype).resolution )
    return recall

def f1(y_true, y_pred):
    

    precision = pp(y_true, y_pred)
    recall = se(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_dataset_size(train_list_fn):

    train_samples = 0.0
    win_size_samples = 0.0
    
    if os.path.isfile( train_list_fn ):
        
        aux_df = []
        
        try:
            aux_df = read_csv(train_list_fn, header=None, index_col=False, sep=',' )

            paths = aux_df[0].values
            paths = paths.tolist()
            
            train_samples = np.sum( aux_df[1].values )
            win_size_samples = aux_df[2].values
            win_size_samples = win_size_samples[0]
            
        except:
            
            try:
                paths = np.loadtxt(train_list_fn, dtype=list ).tolist()
            except:
                paths = glob(train_list_fn)
            
    else:
        paths = glob(train_list_fn)
    
    if not isinstance(paths, list):
        paths = [paths]

    if train_samples == 0.0:
        
        if len(paths) == 0:
            raise EnvironmentError
        else:
            cant_train_parts = len(paths)
            for ii in range(cant_train_parts):
                train_ds = np.load( paths[ii] )[()]
                train_samples += train_ds['cant_total_samples']
            
            win_size_samples = (train_ds['signals']).shape[1]
    else:
        train_ds = np.load( paths[0] )[()]
            
    return train_samples, win_size_samples, paths

def my_int(x):
    
    return int(np.round(x))

def my_ceil(x):
    
    return int(np.ceil(x))

def lr_sched( ii, this_lr ):
    
    if ii > 0 and (my_int(ii) % 10) == 0 :

        new_lr = this_lr * 0.8;
        
    else:
        
        new_lr = this_lr;

    return(new_lr)
    
def check_datasets( data_gen ) :

    
    (train_x, train_y) = next(data_gen)
    sample_size = 20
    ## Debug signals in train and val sets
    plt.figure(1); idx = np.random.choice(np.array((train_y==1).nonzero()).flatten(), sample_size, replace=False ); sigs = train_x[idx,:,0] ;plt.plot(np.transpose(sigs)); plt.title( str(len(sigs)) +  ' Latidos'); plt.ylim((-(2**15-1), 2**15-1)); plt.show();
    plt.figure(1); idx = np.random.choice(np.array((train_y==0).nonzero()).flatten(), sample_size, replace=False ); sigs = train_x[idx,:,0] ;plt.plot(np.transpose(sigs)); plt.title( str(len(sigs)) +  ' NO Latidos'); plt.ylim((-(2**15-1), 2**15-1)); plt.show();
    #
    #plt.figure(1); idx = np.random.choice(np.array((val_y==1).nonzero()).flatten(), 100, replace=False ); sigs = val_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))
    #plt.figure(1); idx = np.random.choice(np.array((val_y==0).nonzero()).flatten(), 100, replace=False ); sigs = val_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))    


def my_delta_time( t_secs ):
    
    t_hour_delta = t_secs // (60 * 60)
    t_rem = t_secs % (60 * 60)
    t_min_delta = t_rem // 60
    t_sec_delta = t_rem % 60
    
    return '{:2.0f}:{:2.0f}:{:3.3f}'.format(t_hour_delta, t_min_delta, t_sec_delta)

def define_model( model_params ) :
    
    cant_cnn = model_params['cant_cnn']
    cant_filtros = model_params['cant_filtros']
    size_filtros = model_params['size_filtros']
    hidden_dims  = model_params['hidden_dims']
    drop_out = model_params['drop_out']
    
    
    with tf.device('/GPU:0'):
#    with tf.device('/CPU:0'):
        model = Sequential()
    
        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        model.add(Conv1D(cant_filtros,
                         size_filtros,
                         input_shape=(train_features, 1),
                         strides=1,
                         padding='valid',
                         activation='relu'
                         ))
    #    model.add(MaxPooling1D(pool_size=2,
    #                           strides = 2))
        model.add(BatchNormalization())
        
        for _ in range(cant_cnn-1) :
        
            model.add(Conv1D(cant_filtros,
                             size_filtros, 
                             padding='valid'))
            
            model.add(BatchNormalization())
        
        
    #    model.add(Dropout(0.25))
        
        # we use max pooling:
        model.add(GlobalMaxPooling1D())
        
        # We add a vanilla hidden layer:
        model.add(Dense(2*hidden_dims))
        model.add(Activation('relu'))
        model.add(Dropout(drop_out))
        
        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dims))
        model.add(Activation('relu'))
        model.add(Dropout(drop_out))
        
        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=this_lr),
                      metrics=[t_f1, t_pp, t_se])
        
        return(model)    

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

parser.add_argument( '--learning_rates', 
                     default=[], 
                     nargs="*",
                     type=float, 
                     help='Nombre de la base de datos')

parser.add_argument( '--dropout', 
                     default=0.25, 
                     type=float, 
                     help='Nombre de la base de datos')

parser.add_argument( '--batch_size', 
                     default=2 ** 10, 
                     type=int, 
                     help='Nombre de la base de datos')

parser.add_argument( '--epochs', 
                     default=10, 
                     type=int, 
                     help='Nombre de la base de datos')

args = parser.parse_args()

# data
train_list_fn = args.train_list
val_list_fn = args.val_list
test_list_fn = args.test_list

# model
drop_out = args.dropout

# Fit configuration
all_lr = np.array(args.learning_rates)
batch_size = args.batch_size
epochs = args.epochs


## Train
print('Build train generator ...')

if train_list_fn == '':
    raise EnvironmentError

train_samples, train_features, train_paths = get_dataset_size(train_list_fn)

train_generator = generator_class(train_paths, batch_size)


if val_list_fn == '':
    
    val_generator = []
    
else:
    
    print('Build val generator ...')
    
    val_samples, val_features, val_paths = get_dataset_size(val_list_fn)
    
    val_generator = generator_class(val_paths, batch_size)


test_generator = []

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


#bDebug = True
bDebug = False

if bDebug :
    
    check_datasets( train_generator ) 
    check_datasets( val_generator ) 

model_params = { 'cant_cnn': 16,
                 'cant_filtros': 48,
                 'size_filtros': 3,
                 'hidden_dims': 48,
                 'drop_out': drop_out}

if all_lr.size == 0 :
    
    all_lr = np.logspace(-5, -3, 5)



for this_lr in all_lr :
    
    model = define_model(model_params)
    
    print('LR: ' + str(this_lr) )
    print('##########')
    
    #my_callback = MyCallbackClass()
    
    print('Start training @ ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    start_time = time.time()
    
    train_steps = my_ceil(train_samples / batch_size)
    val_steps = my_ceil(val_samples / batch_size)
    
    
    
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = train_steps,
                                  epochs = epochs,
                                  validation_data = val_generator,
                                  validation_steps = my_int(val_steps/4),
                                  callbacks=[ TerminateOnNaN(),
                                              EarlyStopping(
                                                           monitor='val_t_f1', 
                                                           min_delta=0.001, 
                                                           patience=10, 
                                                           mode='max', 
                                                           restore_best_weights=True),
                                              LearningRateScheduler(lr_sched, verbose=1)
                                            ]
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
    
    
    print('End training @ ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
    time_elapsed = time.time() - start_time

    print( 'Time elapsed to train: ' + my_delta_time(time_elapsed) )
    
    result_path = os.path.join('.', 'results')
    os.makedirs(result_path, exist_ok=True)
    
    model_id = time.strftime("%d_%b_%Y_%H_%M_%S", time.localtime()) + '_lr_{:3.3g}'.format(this_lr) 
    model.save( os.path.join( result_path, model_id + 'qrs_detector_model_' + '.h5'))  # creates a HDF5 file 'my_model.h5'
    np.save( os.path.join( result_path, model_id + '_history.npy'), {'history' : history})

    bWithTestEval = False

    if bWithTestEval :
        
        print('Build test generator ...')
           
        if test_generator == [] :
            
            if test_list_fn == '' :
                
                test_steps = 0
                
            else:
                
                test_samples, test_features, test_paths = get_dataset_size(test_list_fn)
                
                test_generator = generator_class(test_paths, batch_size)
                
                test_steps = my_ceil(test_samples / batch_size)


    
    #train_se = history.history['se']
    ##val_se = my_callback.val_recalls
    #
    #train_pp = history.history['pp']
    ##val_pp = my_callback.val_precisions
    #
    train_f1 = history.history['t_f1']
    val_f1 = history.history['val_t_f1']
    val_se = history.history['val_t_se']
    val_pp = history.history['val_t_pp']
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Create count of the number of epochs
    epoch_count = range(1, len(val_loss) + 1)

    print('LR: ' + str(this_lr) )
    print('##########')

    train_eval = model.evaluate_generator(
                            train_generator,
                            steps = train_steps
                            )
    
    
    aux_str = 'Train\n-----\n'
    for (metric_name, metric_val) in zip(model.metrics_names, train_eval):
        aux_str += '{}: {:3.3g} '.format(metric_name, metric_val)
    aux_str += '\n'
    print(aux_str)
    
    val_eval = model.evaluate_generator(
                            val_generator,
                            steps = val_steps
                            )
    
    
    aux_str = 'Val\n-----\n'
    for (metric_name, metric_val) in zip(model.metrics_names, val_eval):
        aux_str += '{}: {:3.3g} '.format(metric_name, metric_val)
    aux_str += '\n'
    print(aux_str)
    
    
    if bWithTestEval :
    
        if test_generator == [] :
    
            aux_str = 'Diferencia\n-------------\n'
            for (metric_name, train_val, test_val) in zip(model.metrics_names, train_eval, val_eval):
                aux_str += '{}: {:3.3g} '.format(metric_name, train_val - test_val)
            aux_str += '\n'
            print(aux_str)
    
        else:
            
            test_eval = model.evaluate_generator(
                                    test_generator,
                                    steps = test_steps
                                    )
            
            
            aux_str = 'Test\n-----\n'
            for (metric_name, metric_val) in zip(model.metrics_names, test_eval):
                aux_str += '{}: {:3.3g} '.format(metric_name, metric_val)
            aux_str += '\n'
            print(aux_str)
                
            
            aux_str = 'Diferencia\n-------------\n'
            for (metric_name, train_val, test_val) in zip(model.metrics_names, train_eval, test_eval):
                aux_str += '{}: {:3.3g} '.format(metric_name, train_val - test_val)
            aux_str += '\n'
            print(aux_str)
        
    else:
        
        test_eval = [np.nan] * len(train_eval)


    fig_hdl, axes = plt.subplots(2, 1, clear=True, sharex = True)

    # Visualize F1 history
    axes[0].plot(epoch_count, np.transpose(np.array((train_f1, val_f1, val_se, val_pp))))
    axes[0].set_ylim([0.9, 1])
    axes[0].legend(['train_f1', 'val_f1', 'val_se', 'val_pp' ])
    axes[0].set_ylabel('F Score')
    axes[0].set_title('F1 ' + '_lr_{:3.3g} '.format(this_lr) );
    
    axes[1].plot(epoch_count, train_loss, 'r--')
    axes[1].plot(epoch_count, val_loss, 'b-')
    axes[1].set_ylim([0, np.max(np.vstack([train_loss, val_loss]))])
    axes[1].legend(['Train', 'Val'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss' + '_lr_{:3.3g} '.format(this_lr) );
        
#    aux_metrics = [ [train_eval[ii], val_eval[ii], test_eval[ii]]  for ii in range(1,len(model.metrics_names)) ]
#    aux_metrics = np.transpose(np.array(aux_metrics))
#    
#    axes[0,1].plot(range(3), aux_metrics, 'o--' )
#    axes[0,1].set_xticks(np.arange(3), ('Train', 'Val', 'Test'))
#    axes[0,1].legend(model.metrics_names[1:])
#    axes[0,1].set_title('Métricas en los datasets ' + '_lr_{:3.3g} '.format(this_lr) );
#    
#    
#    axes[1,1].plot(range(3), np.array([train_eval[0], val_eval[0], test_eval[0]]) )
#    axes[1,1].set_xticks(np.arange(3), ('Train', 'Val', 'Test'))
#    axes[1,1].legend(['Loss'])
#    axes[1,1].set_title('Loss en los datasets ' + '_lr_{:3.3g} '.format(this_lr) );
    
    plt.savefig(os.path.join( result_path, model_id + '.jpg'), dpi=300)

