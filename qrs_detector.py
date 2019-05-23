#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:21:32 2019

@author: mariano
"""

# Importo TensorFlow como tf
#import tensorflow as tf
# Importo keras
#import keras as kr

# Librerias auxiliares
import numpy as np
import sys
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt

from fractions import Fraction
from pandas import DataFrame, read_csv
#from IPython.display import HTML
import os
from glob import glob
#import h5py
import wfdb as wf
from scipy import signal as sig
import argparse as ap
from statsmodels.robust.scale import mad
import re


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

def my_int(x):
    
    return int(np.round(x))

def my_ceil(x):
    
    return int(np.ceil(x))

def center_data(x):
    
    return( x - np.nanmedian(x) )

def refine_location( x, my_win):
    
    detection_sig = np.abs(sig.filtfilt(my_win,1, x, axis=0 ))

    return( np.argmax(detection_sig) )


parser = ap.ArgumentParser(description='Deep learning based QRS detector')

parser.add_argument( 'recording', 
                     default = '', 
                     type=str, 
                     help='Path to an ECG recording or to a folder with recordings.')

parser.add_argument( '--detector', 
                     default='', 
                     type=str, 
                     help='Path to a trained QRS detector model.')

args = parser.parse_args()

recording_fn = args.recording
detector_fn = args.detector

if os.path.isfile(recording_fn):
    # recname with extension
    records = os.path.split(recording_fn) 
    rec_path = records[0]
    records = [ records[1][:-4] ]

elif os.path.isfile(os.path.join(recording_fn, '.hea')):
    # recname without extension
    records = os.path.split(recording_fn) 
    rec_path = records[0]
    records = recording_fn
    
elif os.path.isdir(recording_fn):
    # all recordings in folder 
    
    paths = glob(os.path.join(recording_fn, '*.hea'))

    if len(paths) == 0:
        print( 'No header file found (*.hea) in: {:s}'.format(recording_fn) )
        sys.exit(1)
        
    # Elimino la extensión
    rec_path = recording_fn
    paths = [os.path.split(path) for path in paths]
    file_names = [path[1][:-4] for path in paths]
    file_names.sort()
    
    records = file_names
    
else:
    # error
    print( 'Must be a folder or file: {:s}'.format(recording_fn) )
    sys.exit(1)
    

ds_config = { 
                'width':     .2, # (s) Width of the Deep Neural Network (DNN)
                'distance':  .3, # (s) Minimum separation between consequtive QRS complexes
                'explore_win': 60, # (s) window to seek for possible heartbeats to calculate scales and offset
                'target_fs': 250, # (Hz) Internal sampling rate. 
             } 


# load model

if os.path.isfile(detector_fn):

    dependencies = {
         't_f1': t_f1,
         't_pp': t_pp,
         't_se': t_se
    }
    
    model = load_model(detector_fn, custom_objects=dependencies)

else:
    # error
    print( 'Model file not found: {:s}'.format(detector_fn) )
    sys.exit(1)

# prepare data to feed DNN

dist_in_samp = my_int( ds_config['distance'] * ds_config['target_fs'])
w_in_samp = my_int( ds_config['width'] * ds_config['target_fs'])
hw_in_samp = my_int( ds_config['width'] * ds_config['target_fs'] / 2)

my_win = sig.gaussian(hw_in_samp+1, (hw_in_samp-1)/5 )
my_win = np.diff(my_win) * sig.gaussian(hw_in_samp, (hw_in_samp-1)/5 )

for this_rec in records:

    data, field = wf.rdsamp(os.path.join(rec_path, this_rec) )
    
    pq_ratio = ds_config['target_fs']/field['fs']
    resample_frac = Fraction( pq_ratio ).limit_denominator(20)
    #recalculo el ratio real
    pq_ratio = resample_frac.numerator/ resample_frac.denominator
    data = sig.resample_poly(data, resample_frac.numerator, resample_frac.denominator )
    
    detection_sig = np.abs(sig.filtfilt(my_win,1, data[:my_int(np.min([data.shape[0], ds_config['explore_win'] * ds_config['target_fs'] ])) , :], axis=0 ) )
    
    this_scale = np.repeat(np.nan, field['n_sig']).reshape(field['n_sig'], 1)

    for ii in range(field['n_sig']):
        posible_beats, _ = sig.find_peaks(detection_sig[:,ii], height = np.percentile(detection_sig[:,ii], 60) , distance = dist_in_samp)
        posible_beats, _ = sig.find_peaks(detection_sig[:,ii], height = np.mean(detection_sig[posible_beats,ii])/2 , distance = dist_in_samp)

#        plt.figure(1); plt.plot(detection_sig[np.max([0, posible_beats[0]-w_in_samp]):posible_beats[-1]+w_in_samp, ii]); plt.plot( posible_beats, detection_sig[posible_beats, ii] , 'rx'); plt.show(1);        
        
        # scale of the whole recording
        this_scale[ii] = np.nanmedian(np.vstack([ np.max(np.abs(center_data(data[my_int(np.max([0, this_beat-w_in_samp])):my_int(np.min([field['sig_len'] * pq_ratio, this_beat+w_in_samp])) , ii]) ), axis = 0 ) for this_beat in posible_beats ]), axis = 0) 

#        plt.figure(1); plt.plot(data[np.max([0, posible_beats[0]-w_in_samp]):posible_beats[-1]+w_in_samp, ii]); plt.plot( posible_beats, data[posible_beats, ii] , 'rx'); plt.show(1);
             
    seq_index = np.arange(0, data.shape[0], w_in_samp )
    
    ldata_reshaped = len(seq_index)
    
    data_reshaped = data[:ldata_reshaped*w_in_samp,:].reshape(w_in_samp, field['n_sig'] * ldata_reshaped, order='F')

    # unbias and scale
    data_reshaped = data_reshaped - np.nanmedian(data_reshaped, axis=0, keepdims = True)
    
    data_reshaped = data_reshaped * np.repeat(1/this_scale, ldata_reshaped ).reshape(1, ldata_reshaped * field['n_sig'])

    data_reshaped = np.clip(np.round(data_reshaped * (2**15-1) * 0.5), -(2**15-1), 2**15-1 )

    # predict
#    plt.figure(1); idx = np.random.choice(ldata_reshaped * field['n_sig'], 50, replace=False ); plt.plot(data_reshaped[:,idx]); plt.plot( np.repeat(-(2**15-1), data_reshaped.shape[0]), 'k--');  plt.plot( np.repeat((2**15-1), data_reshaped.shape[0]), 'k--'); plt.show(1);

#    plt.figure(1); plt.plot( data_reshaped[:,0] ); plt.show(1);
    
    data_reshaped = np.transpose(data_reshaped).reshape(data_reshaped.shape[1], data_reshaped.shape[0], 1, order='f')
    
#    plt.figure(1); plt.plot( pepe[10, :, :] ); plt.show(1);
    
    predictions = np.round(model.predict( data_reshaped ))
#    predictions = predictions.reshape(ldata_reshaped, field['n_sig'], order='F')

    qrs_idx = np.array((predictions.flatten() == 1).nonzero()).flatten()
    qrs_refine = np.array([refine_location( np.squeeze(data_reshaped[ii,:,:]), my_win )  for ii in qrs_idx ])

    refine_location( np.squeeze(data_reshaped[qrs_idx[0],:,:]), my_win ) 

    seq_index = np.tile(seq_index, field['n_sig'])
    qrs_locations = np.zeros(seq_index.shape)
    qrs_locations[qrs_idx] = np.round( (seq_index[qrs_idx] + qrs_refine) * 1/pq_ratio )

    qrs_locations = qrs_locations.reshape(ldata_reshaped, field['n_sig'], order='F')
    
    qrs_loc_dict = { lead_name: np.sort(qrs_locations[qrs_locations.nonzero(),ii]) for (ii, lead_name) in zip(range(field['n_sig']), field['sig_name']) }
    
    qrs_loc_dict['fs'] = field['fs']
    
    predictions = predictions.reshape(ldata_reshaped, field['n_sig'], order='F')
    
    plt.figure(1); sample_size = 30; idx = np.random.choice(np.array((predictions==1).nonzero()).flatten(), sample_size, replace=False ); sigs = np.transpose(np.squeeze(data_reshaped[idx,:,:])); plt.plot(sigs); plt.show(1);
    



