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

parser = ap.ArgumentParser(description='Prueba para entrenar un detector de QRS mediante técnicas de deep learning')
parser.add_argument( 'db_path', 
                     default='/home/mariano/mariano/dbs/mitdb', 
                     type=str, 
                     help='Path a la base de datos')


args = parser.parse_args()


data_path = args.db_path


def get_records( data_path ):

    # Descargo si no existe
    if os.path.isdir( data_path ):
            
        # Hay 3 archivos por record
        # *.atr es uno de ellos
        paths = glob(os.path.join(data_path, '*.atr'))
    
        # Elimino la extensión
        paths = [os.path.split(path) for path in paths]
        file_names = [path[1][:-4] for path in paths]
        file_names.sort()
        records = file_names

    else:
        
        records = []

    return records

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

def convert_input(channel, beats):
    # Me quedo con todo los latidos

    # Creo una señal con deltas en los latidos
    dirac = np.zeros_like(channel)
    dirac[beats] = 1.0

    # Uso la ventana de hamming para la campana
    width = 36
    filter = sig.hamming(width)
    gauss = np.convolve(filter, dirac, mode = 'same')

    return dirac, gauss

def my_int(x):
    
    return int(np.round(x))

def make_dataset(records, data_path, ds_config, data_aumentation = 1):

    signals, labels = [], []

    nQRS_QRS_ratio = []
    cant_latidos_total = 0

    # Recorro los archivos
    for this_rec in records:
        
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
        this_lab = np.concatenate( ( np.zeros( ( my_int(len(no_QRS_ranges) * field['n_sig']) ,1) ), np.ones( (my_int(len(QRS_ranges) * field['n_sig']), 1) ) ), axis = 0 )
        
        # unbias and normalize
        bScaleRecording = True
#        bScaleRecording = False
        if bScaleRecording:
            this_scale = mad(data, axis=0).reshape(field['n_sig'], 1 )
        
        starts = np.vstack(starts)
        the_sigs = []
        for this_start in starts :
        
            try:
                this_sig = np.transpose(data[my_int(this_start):my_int(this_start + w_in_samp), :]) 
                # unbias and normalize
                this_sig = this_sig - np.median(this_sig, axis=1, keepdims = True)
                
                if not(bScaleRecording):
                    this_scale = mad(this_sig, center = 0, axis=1 ).reshape(this_sig.shape[0],1 )
                    
                this_sig = this_sig * 1/this_scale
                
                the_sigs += [this_sig]
            
            except Exception:
                
                a = 0
                
                
        
        if len(signals) == 0:
            all_labels = this_lab
            all_signals = np.vstack(the_sigs)
        else:
            all_labels = np.concatenate( (all_labels, this_lab) )
            all_signals = np.concatenate( (all_signals, np.vstack(the_sigs)) )
        
        # convierto las anotaciones según el modelo
#        if ds_config['mode'] == 'binary':
#    
#            
#        elif ds_config['mode'] == 'categorical':
#            
#        else:

        # Acumulo

    return all_signals, all_labels
#    # Guardo en forma de diccionario
#    np.save(savepath, {'signals' : signals,
#                       'labels'  : labels })

def convert_data(data, annotations, ds_config):
    
    signals, labels = [], []
    

        
        
    dirac = np.zeros((data.shape[0],1))
    dirac[beats] = 1.0
    
    signals = data
    
    return signals, dirac

#    # Convierto ambos canales
#    for it in range(data.shape[1]):
#        channel = data[:, it]
#        dirac, gauss = convert_input(channel, beats)
#        # Junto los labesl
#        label = np.vstack([dirac, gauss])
#
#        # Ventana movil
#        sta = 0
#        end = width
#        stride = width
#        while end <= len(channel):
#            # Me quedo con una ventana
#            s_frag = channel[sta : end]
#            l_frag = label[:, sta : end]
#
#            # Acumulo
#            signals.append(s_frag)
#            labels.append(l_frag)
#
#            # Paso a la ventana siguiente
#            sta += stride
#            end += stride
#
#    # Convierto a np.array
#    signals = np.array(signals)
#    labels = np.array(labels)
#
#    return signals, labels


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

train_filename = './train.npy'
test_filename = './test.npy'
val_filename = './val.npy'

#bRedo_ds = True
bRedo_ds = False

if  not os.path.isfile( train_filename ) or bRedo_ds:

    # Preparo los archivos
    record_names = get_records(data_path)
    
    # debug
    #record_names = record_names[0:9]
    
    cant_records = len(record_names)
    print( 'Encontramos ' + str(cant_records) + ' registros' )
    
    # particionamiento de 3 vías 
    # train      80%
    # validation 20%
    # eval       20%
    record_names = np.unique(record_names)
    train_recs = np.random.choice(record_names, int(cant_records * 0.8), replace=False )
    test_recs = np.setdiff1d(record_names, train_recs, assume_unique=True)
    val_recs = np.random.choice(train_recs, int(cant_records * 0.2), replace=False )
    train_recs = np.setdiff1d(train_recs, val_recs, assume_unique=True)
    
    
    # Armo el set de entrenamiento, aumentando para que contemple desplazamientos temporales
    train_ds = make_dataset(train_recs, data_path, ds_config, data_aumentation = 5 )
    
    # Armo el set de validacion
    val_ds = make_dataset(val_recs, data_path, ds_config)
    
    # Armo el set de testeo
    test_ds = make_dataset(test_recs, data_path, ds_config)
    
    np.save(train_filename, {'recordings' : train_recs, 'signals' : train_ds[0], 'labels'  : train_ds[1]})
    np.save(test_filename,  {'recordings' : test_recs,  'signals' : test_ds[0],  'labels'  : test_ds[1]})
    np.save(val_filename,   {'recordings' : val_recs,   'signals' : val_ds[0],   'labels'  : val_ds[1]})

else:
    
    train_ds = np.load(train_filename)
    val_ds = np.load(val_filename)
    
    print('Build model...')
    model = Sequential()
    
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    
    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    