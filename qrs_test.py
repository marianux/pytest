#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:30:03 2019

@author: mariano
"""

# Importo TensorFlow como tf
#import tensorflow as tf
# Importo keras
#import keras as kr

# Librerias auxiliares
import numpy as np
import matplotlib.pyplot as plt
#from pandas import DataFrame
#from IPython.display import HTML
import os
from glob import glob
import h5py
import wfdb as wf
from scipy import signal as sig
import argparse as ap
from statsmodels.robust.scale import mad

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Conv1D, GlobalMaxPooling1D, MaxPooling1D

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import backend as K
from keras.optimizers import Adam

 
parser = ap.ArgumentParser(description='Prueba para entrenar un detector de QRS mediante técnicas de deep learning')
parser.add_argument( 'db_path', 
                     default='/home/mariano/mariano/dbs/', 
                     type=str, 
                     help='Path a la base de datos')

parser.add_argument( '--db_name', 
                     default='', 
                     type=str, 
                     help='Nombre de la base de datos')

args = parser.parse_args()

db_path = args.db_path
db_name = args.db_name

if db_name == '':
    # default databases
    db_name = ['stdb', 'INCART', 'mitdb' ,'ltdb' ,'E-OTH-12-0927-015' ,'ltafdb' ,'edb' ,'aha' ,'sddb' ,'svdb' ,'nsrdb' ,'ltstdb' , 'biosigna']
    


class MyCallbackClass(Callback):
    
    def on_train_begin(self, logs={}):
     self.val_f1s = []
     self.val_recalls = []
     self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
     val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
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


def get_records( db_path, db_name ):

    all_records = []
    all_patient_list = []
    size_db = []
    
#    if db_name=='':
#        # explore all databases        
#        
#        db_names = [ name for name in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, name)) ]
#
#    else:
#        
#        db_names = db_name

    for this_db in db_name:

        records = []
        patient_list = []

        # particularidades de cada DB
#            if this_db == 'mitdb':
#                
#                records = ['100', '101', '103', '105', '106', '108', '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
#                patient_list = np.arange(0, len(records)) + 1
#                
#                
#                
#            elif this_db == 'svdb':
#                records = ['800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810', '811', '812', '820', '821', '822', '823', '824', '825', '826', '827', '828', '829', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852', '853', '854', '855', '856', '857', '858', '859', '860', '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '890', '891', '892', '893', '894']
#                patient_list = np.arange(0, len(records)) + 1
#                
#            elif this_db == 'INCART':
        if this_db == 'INCART':
            # INCART: en esta DB hay varios registros por paciente
            records = [ 'I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08', 'I09', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20', 'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I29', 'I30', 'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40', 'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'I50', 'I51', 'I52', 'I53', 'I54', 'I55', 'I56', 'I57', 'I58', 'I59', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'I70', 'I71', 'I72', 'I73', 'I74', 'I75']
    
            patient_list = np.array([
                             1, 1 ,    # patient 1
                             2, 2, 2 , # patient 2
                             3, 3 ,    # ...
                             4 ,       #
                             5, 5, 5 , #
                             6, 6, 6 , #
                             7 , #
                             8, 8 , #
                             9, 9 , #
                             10, 10, 10 , #
                             11, 11 , #
                             12, 12 , #
                             13, 13 , #
                             14, 14, 14, 14 , #
                             15, 15 , #
                             16, 16, 16 , #
                             17, 17 , #
                             18, 18 , #
                             19, 19 , #
                             20, 20, 20 , #
                             21, 21 , #
                             22, 22 , #
                             23, 23, 23 , #
                             24, 24, 24 , #
                             25, 25 , #
                             26, 26, 26 , #
                             27, 27, 27 , #
                             28, 28, 28 , #
                             29, 29 , #
                             30, 30 , #
                             31, 31 , # ...
                             32, 32   # patient 32
                             ])
        else:
            
            # para el resto de dbs, un registro es un paciente.                        
            data_path = os.path.join(db_path, this_db)
            
            paths = glob(os.path.join(data_path, '*.atr'))
        
            if paths == []:
                continue
                
            # Elimino la extensión
            paths = [os.path.split(path) for path in paths]
            file_names = [path[1][:-4] for path in paths]
            file_names.sort()
            
            records = file_names
            patient_list = np.arange(0, len(records)) + 1


        print( 'Procesando ' + this_db )

        records = [os.path.join(this_db, this_rec) for this_rec in records ]
        
        size_db += [ len(np.unique(patient_list)) ]
        all_records += records
        all_patient_list = np.hstack( [ all_patient_list, (patient_list + len(np.unique(all_patient_list)) ) ] )


    return all_records, all_patient_list, size_db

def get_beats(annotation):

    no_beats = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']
    
    not_beats_mk = np.isin(annotation.symbol, no_beats, assume_unique=True)
    beats_mk = np.logical_and( np.ones_like(not_beats_mk), np.logical_not(not_beats_mk) )

    # Me quedo con las posiciones
    beats = annotation.sample[beats_mk]

    return beats

def gen_interest_ranges( start_end, references, width):
    
    no_QRS_ranges, QRS_ranges = [], []
    
    nrefs = len(references)
    
    if nrefs > 0 and len(start_end) > 0 :
        
        references = np.unique(references)
        start_end.sort()

        half_win = int(np.round(width/2))

        references = references[np.logical_and(references >= width, references <= (start_end[1] - width ))]
        nrefs = len(references)
        
        this_start = start_end[0]
        this_end = references[0] - half_win - 1
        st_idx = 0
        
        QRS_ranges = []
        no_QRS_ranges = []
        
#        while st_idx < nrefs:
#
#            if this_end - width > this_start:
#                
#                no_QRS_ranges += [ ( this_start , this_end )  ]
#
#                this_qstart = references[st_idx] - half_win
#                this_qend = this_qstart + width
#                QRS_ranges += [ ( this_qstart, this_qend )  ]
#                
#                break
#            
#            else:
#
# lo descarto porque no me interesa que pudiera ser más corto que  width               
#                this_qstart = np.max( (0, references[st_idx] - half_win) )
#                this_qend = this_qstart + width
#                QRS_ranges += [ ( this_qstart, this_qend )  ]
                
#                this_start = this_end + width + 2
#                this_end = references[st_idx+1] - half_win - 1
#                st_idx += 1
        
        for ii in np.arange(st_idx, nrefs):
            
            this_start = this_end + width + 2
            this_end = references[ii] - half_win - 1
            
            if this_end - width > this_start:
                
                no_QRS_ranges += [ ( this_start , this_end )  ]

            this_qstart = references[ii] - half_win
# con numpy clipea automáticamente
#            this_qend = np.min( ( start_end[1], this_qstart + width) )
            this_qend = this_qstart + width
            QRS_ranges += [ ( this_qstart, this_qend )  ]

#        # última referencia
#        this_start = this_end + width + 2
#        this_end = references[-1] - half_win - 1
#        
#        if this_end - width > this_start:
#            
#            no_QRS_ranges += [ ( this_start , this_end )  ]
#    
#        # posible último tramo
#        this_start = this_end + width + 2
#        this_end = start_end[1]
#        
#        if this_end - width > this_start:
#            
#            no_QRS_ranges += [ ( this_start , this_end )  ]
        
    
    return no_QRS_ranges, QRS_ranges

def my_int(x):
    
    return int(np.round(x))

def make_dataset(records, data_path, ds_config, data_aumentation = 1):

    signals, labels = [], []

    nQRS_QRS_ratio = []
    cant_latidos_total = 0

    # Recorro los archivos
#    for this_rec in records:
    
    aux_idx = ((np.array(records) == 'stdb/315').nonzero())[0][0]
    
    for ii in np.arange(aux_idx, len(records)):
        
        this_rec = records[ii]
        print ('Procesando:' + this_rec)
        data, field = wf.rdsamp(os.path.join(data_path, this_rec) )
        annotations = wf.rdann(os.path.join(data_path, this_rec), 'atr')

        # filtro los latidos 
        beats = get_beats(annotations)

        cant_latidos_total += len(beats)

        w_in_samp = my_int( ds_config['width'] * field['fs'])
        hw_in_samp = my_int( ds_config['width'] * field['fs'] / 2)
        tol_in_samp = my_int( ds_config['heartbeat_tolerance'] * field['fs'])
        samp_around_beats = w_in_samp * len(beats)
        # acumulo los ratios de QRS sobre el total de muestras para mantener ese ratio en el train ds
        this_ratio = (field['sig_len'] - samp_around_beats)/samp_around_beats
        nQRS_QRS_ratio.append(this_ratio)

    # target proportion ratio 
    tgt_ratio = np.median(nQRS_QRS_ratio)
    
    signals = []
    labels = []

    for this_rec in records:

        print ('Procesando:' + this_rec)
        data, field = wf.rdsamp(os.path.join(data_path, this_rec) )
        annotations = wf.rdann(os.path.join(data_path, this_rec), 'atr')
        
        # filtro los latidos 
        beats = get_beats(annotations)

        # genero las referencias temporales para generar los vectores de entrada
        no_QRS_ranges, QRS_ranges = gen_interest_ranges( start_end=[0, field['sig_len']], references = beats, width = w_in_samp )

        # aumento la cantidad de segmentos no_QRS de acuerdo al ratio deseado 
        segments_repeat = my_int(np.abs( len(no_QRS_ranges) - np.ceil( len(QRS_ranges) * tgt_ratio ) ))
        no_QRS_ranges += [ no_QRS_ranges[np.random.randint(len(no_QRS_ranges))] for _ in range(segments_repeat) ]
        
        # genero los comienzos aumentados de acuerdo a data_aumentation
        starts = []
        
        starts += [ (np.random.randint(this_start_end[0] - hw_in_samp, this_start_end[1] - hw_in_samp, size=data_aumentation, dtype='int')).reshape(data_aumentation,1) for this_start_end in no_QRS_ranges ]
        starts += [ (np.random.randint(this_start_end[0] - tol_in_samp, this_start_end[0] + tol_in_samp, size=data_aumentation, dtype='int')).reshape(data_aumentation,1) for this_start_end in QRS_ranges ]
#        starts += [ this_start_end[0] for this_start_end in QRS_ranges ]
        
        # 0: No QRS - 1: QRS
        this_lab = np.concatenate( ( np.zeros( ( my_int(len(no_QRS_ranges) * field['n_sig'] * data_aumentation) ,1) ), np.ones( (my_int(len(QRS_ranges) * field['n_sig'] * data_aumentation), 1) ) ), axis = 0 )
        
        # unbias and normalize
        bScaleRecording = True
#        bScaleRecording = False
        if bScaleRecording:
            this_scale = mad(data, axis=0).reshape(field['n_sig'], 1 )
            bAux = np.bitwise_or( this_scale == 0,  np.isnan(this_scale))
            if np.any(bAux):
                # avoid scaling in case 0 or NaN
                this_scale[bAux] = 1
                
                
        starts = np.vstack(starts)
        the_sigs = []
        for this_start in starts :
        
            try:
                this_sig = np.transpose(data[my_int(this_start):my_int(this_start + w_in_samp), :]) 
                # unbias and normalize
                this_sig = this_sig - np.median(this_sig, axis=1, keepdims = True)
                
                if not(bScaleRecording):
                    this_scale = mad(this_sig, center = 0, axis=1 ).reshape(this_sig.shape[0],1 )
                    bAux = np.bitwise_or( this_scale == 0,  np.isnan(this_scale))
                    if np.any(bAux):
                        # avoid scaling in case 0 or NaN
                        this_scale[bAux] = 1
                    
                # add an small dither 
                this_sig = this_sig * 1/this_scale + 1/500 * np.random.randn(this_sig.shape[0], this_sig.shape[1])
                
                the_sigs += [this_sig]
            
            except Exception:
                
                a = 0
                
        
        if len(signals) == 0:
            all_labels = this_lab
            all_signals = np.vstack(the_sigs)
        else:
            all_labels = np.concatenate( (all_labels, this_lab) )
            all_signals = np.concatenate( (all_signals, np.vstack(the_sigs)) )
        

    return all_signals, all_labels


ds_config = { 
                'width': .2, # s
                'mode': 'binary', # cada ventana de *w* muestras tendrá latido/no_latido
#                'mode': 'categorical', # cada ventana de *w* muestras tendrá k categorías 
                                # 0: no_latido - 
                                # 1: QRS en la primer ventana de ancho w_1 = w/(ceil(w/150ms))
                                # ...
                                # k-1: QRS en la última ventana w_k
                'heartbeat_tolerance': .07, # s
             } 

train_filename =  'train_' + '_'.join(db_name) + '.npy'
test_filename = 'test_' + '_'.join(db_name) + '.npy'
val_filename = 'val_' + '_'.join(db_name) + '.npy'

#bRedo_ds = True
bRedo_ds = False

if  not os.path.isfile( train_filename ) or bRedo_ds:

    # Preparo los archivos
    record_names, patient_list, size_db = get_records(db_path, db_name)
    
    # debug
    #record_names = record_names[0:9]

    patient_indexes = np.unique(patient_list)
    cant_patients = len(patient_indexes)
#    record_names = np.unique(record_names)
    cant_records = len(record_names)
    
    print( 'Encontramos ' + str(cant_patients) + ' pacientes y ' + str(cant_records) + ' registros.' )
    
    # propocion de cada db en el dataset
    prop_db = size_db / np.sum(size_db)
    
    tgt_train_size = my_int(cant_patients * 0.8)
    
    tgt_db_parts_size = tgt_train_size * prop_db
    
    tgt_db_parts_size = [ my_int(ii) for ii in tgt_db_parts_size]
    
    db_start = np.hstack([ 0, np.cumsum(size_db[:-1]) ])
    db_end = db_start + size_db
    db_idx = np.hstack([np.repeat(ii, size_db[ii]) for ii in range(len(size_db))])
    
    train_recs = []
    val_recs = []
    test_recs = []
    
    for ii in range(len(db_name)):
        
        # particionamiento de 3 vías 
        # train      80%
        # validation 20%
        # eval       20%
        
        aux_idx = (db_idx == ii).nonzero()
        train_patients = np.sort(np.random.choice(patient_indexes[aux_idx], tgt_db_parts_size[ii], replace=False ))
        test_patients = np.sort(np.setdiff1d(patient_indexes[aux_idx], train_patients, assume_unique=True))
        val_patients = np.sort(np.random.choice(train_patients, my_int(tgt_db_parts_size[ii] * 0.2), replace=False ))
        train_patients = np.setdiff1d(train_patients, val_patients, assume_unique=True)
        
        aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in train_patients]).flatten()
        aux_val = [record_names[my_int(ii)] for ii in aux_idx]
        train_recs += aux_val

        aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in val_patients]).flatten()
        aux_val = [record_names[my_int(ii)] for ii in aux_idx]
        val_recs += aux_val

        aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in test_patients]).flatten()
        aux_val = [record_names[my_int(ii)] for ii in aux_idx]
        test_recs += aux_val
    
#    # particionamiento de 3 vías 
#    # train      80%
#    # validation 20%
#    # eval       20%
#    train_recs = np.random.choice(record_names, int(cant_records * 0.8), replace=False )
#    test_recs = np.setdiff1d(record_names, train_recs, assume_unique=True)
#    val_recs = np.random.choice(train_recs, int(cant_records * 0.2), replace=False )
#    train_recs = np.setdiff1d(train_recs, val_recs, assume_unique=True)
    
#    data_path = os.path.join(db_path, db_name)

    # Armo el set de entrenamiento, aumentando para que contemple desplazamientos temporales
    train_ds = make_dataset(train_recs, db_path, ds_config, data_aumentation = 1 )
    np.save(train_filename, {'recordings' : train_recs, 'signals' : train_ds[0], 'labels'  : train_ds[1]})
    
    # Armo el set de validacion
    val_ds = make_dataset(val_recs, db_path, ds_config)
    np.save(val_filename,   {'recordings' : val_recs,   'signals' : val_ds[0],   'labels'  : val_ds[1]})
    
    # Armo el set de testeo
    test_ds = make_dataset(test_recs, db_path, ds_config)
    np.save(test_filename,  {'recordings' : test_recs,  'signals' : test_ds[0],  'labels'  : test_ds[1]})

else:

    cant_filtros = 12
    size_filtros = 3
    hidden_dims  = 6
    batch_size = 16
    epochs = 5
    
    train_ds = np.load(train_filename)[()]
    train_recs = train_ds['recordings']
    train_x = train_ds['signals']
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    train_y = train_ds['labels']
    train_y = train_y.flatten()
    
    val_ds = np.load(val_filename)[()]
    val_recs = val_ds['recordings']
    val_x = val_ds['signals']
    val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1])
    val_y = val_ds['labels']
    val_y = val_y.flatten()
    
## Debug signals in train and val sets
#plt.figure(1); idx = np.random.choice(np.array((train_y==1).nonzero()).flatten(), 20, replace=False ); sigs = train_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))
#plt.figure(1); idx = np.random.choice(np.array((train_y==0).nonzero()).flatten(), 20, replace=False ); sigs = train_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))
#
#plt.figure(1); idx = np.random.choice(np.array((val_y==1).nonzero()).flatten(), 100, replace=False ); sigs = val_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))
#plt.figure(1); idx = np.random.choice(np.array((val_y==0).nonzero()).flatten(), 100, replace=False ); sigs = val_x[idx,0,:] ;plt.plot(np.transpose(sigs)); plt.ylim((-10,10))    
    
    
    print('Build model...')
    
    
    model = Sequential()
   
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(cant_filtros,
                     size_filtros,
                     input_shape=(1, train_x.shape[2]),
                     strides=1,
                     padding='same',
                     activation='relu'
                     ))
    
#    model.add(Conv1D(cant_filtros,
#                     size_filtros,
#                     padding='same',
#                     activation='relu'
#                     ))
    
    model.add(Dropout(0.25))
    
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    
    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=[f1, pp, se])
    
    my_callback = MyCallbackClass()
    
    history = model.fit(train_x, train_y,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(val_x, val_y),
#                          validation_data=(train_x, train_y),
                          callbacks=[my_callback])
    
    model.save('qrs_detector_model.h5')  # creates a HDF5 file 'my_model.h5'
    
    
    train_se = history.history['se']
    val_se = my_callback.val_recalls
    
    train_pp = history.history['pp']
    val_pp = my_callback.val_precisions
    
    train_f1 = history.history['f1']
    val_f1 = my_callback.val_f1s
    
#    train_loss = history.history['loss']
#    val_loss = history.history['val_loss']
    
    # Create count of the number of epochs
    epoch_count = range(1, epochs + 1)
    
    # Visualize accuracy history
        
    # Visualize accuracy history
    plt.figure(1)
    plt.plot(epoch_count, np.transpose(np.array((train_f1, train_se, train_pp, val_f1, val_se, val_pp))))
    plt.legend(['train_f1', 'train_se', 'train_pp', 'val_f1', 'val_se', 'val_pp'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Score')
    plt.title('F1');
    plt.show();

    plt.figure(2)
    plt.plot(epoch_count, train_loss, 'r--')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Train', 'Val'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss');
    plt.show();
    
    
    train_predict = (np.asarray(model.predict(train_x))).round()
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

