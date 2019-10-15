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
import sys
from fractions import Fraction
from pandas import DataFrame, read_csv
#from IPython.display import HTML
import os
from glob import glob
#import h5py
import wfdb as wf
from scipy import signal as sig
import argparse as ap
import re
import matplotlib.pyplot as plt

from qs_filter_design import qs_filter_design
from scipy.signal._peak_finding_utils import (
    _select_by_peak_distance,
)


def get_records( db_path, db_name ):

    all_records = []
    all_patient_list = []
    size_db = []

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


def my_int(x):
    
    return int(np.round(x))

def my_ceil(x):
    
    return int(np.ceil(x))

def range_estimation( x, fs):
    
    win_size = my_int(5 * fs)
    hwin_size = my_int(win_size /2 )
    explore_win = my_int(ds_config['explore_win'] * fs)
    
    idx = np.arange( np.min([win_size, x.shape[0]]) , np.max([ explore_win - win_size, x.shape[0] - win_size, 0 ]), hwin_size )
    
    if len(idx) == 0:
        # short recording
        hwin_size = np.floor(x.shape[0] / 2)
        idx = [hwin_size]
        
    max_abs = [ np.max(np.abs( x[ii-hwin_size:ii+hwin_size, :])) for ii in idx ]
#   plt.figure(1); plt.plot(x[np.max([0, idx[0]-hwin_size]):idx[-1]+hwin_size, :]); plt.plot( idx, max_abs , 'rx:'); plt.show(1);
    
    return( np.nanmedian(max_abs) )


def range_estimation( x, fs):
    
    win_size = my_int(5 * fs)
    hwin_size = my_int(win_size /2 )
    explore_win = my_int(ds_config['explore_win'] * fs)
    
    idx = np.arange( np.min([win_size, x.shape[0]]) , np.max([ explore_win - win_size, x.shape[0] - win_size, 0 ]), hwin_size )
    
    if len(idx) == 0:
        # short recording
        hwin_size = np.floor(x.shape[0] / 2)
        idx = [hwin_size]
        
    max_abs = [ np.max(np.abs( x[ii-hwin_size:ii+hwin_size, :])) for ii in idx ]
#   plt.figure(1); plt.plot(x[np.max([0, idx[0]-hwin_size]):idx[-1]+hwin_size, :]); plt.plot( idx, max_abs , 'rx:'); plt.show(1);
    
    return( np.nanmedian(max_abs) )


def median_baseline_removal( x, fs):
    
    win_size_short = np.round(0.2 * fs)
    win_size_long = np.round(0.6 * fs)
    
    if( (win_size_short % 2) == 0 ):
        win_size_short += 1
        
    if( (win_size_long % 2) == 0 ):
        win_size_long += 1
    
    return( x - np.transpose(np.array([sig.medfilt(sig.medfilt(x[:,ii], my_int(win_size_short) ), my_int(win_size_long) ) for ii in range(x.shape[1]) ]))  )


def zero_crossings( x ):

    zero_crossings = np.where(np.diff(np.signbit(x)))[0]

    return(zero_crossings)

def if_not_empty(x):

    if len(x) > 0:
        return(x)
    else:
        return(np.nan)

def keep_local_extrema(x, peaks, zero_crossings, distance):

    zero_crossings = np.array(zero_crossings).reshape([len(zero_crossings),1])
    zc = np.hstack([zero_crossings[0:-2], zero_crossings[1:-1], zero_crossings[2:]])
    # zc = np.hstack([zero_crossings[0:-1], zero_crossings[1:]])

    local_extrema_weight = np.array([ np.sum(np.abs( if_not_empty(x[peaks[ np.logical_and( peaks > zc[ii,0], peaks < zc[ii,2])]])  )) for ii in range(zc.shape[0]) ])

    aux_idx = np.logical_not(np.isnan(local_extrema_weight)).nonzero()[0]
    zc = zc[aux_idx,1]
    local_extrema_weight = local_extrema_weight[aux_idx]

    keep = _select_by_peak_distance(zc, local_extrema_weight, distance)
    local_extrema = zc[keep]

    return( local_extrema )

def my_find_extrema( x, this_distance = None ):

    peaks_p = [ sig.find_peaks(np.squeeze(x[:,jj]), distance = this_distance )[0]  for jj in range(x.shape[1]) ]
    peaks_n = [ sig.find_peaks(-np.squeeze(x[:,jj]), distance = this_distance )[0]  for jj in range(x.shape[1]) ]
    zeros = [ zero_crossings( np.squeeze(x[:,jj]) ) for jj in range(x.shape[1]) ]

    my_extrema = [ keep_local_extrema(x[:,ii], np.sort( np.hstack([jj,kk])), ll, this_distance)  for (ii, jj, kk, ll) in zip(range(x.shape[1]), peaks_p, peaks_n, zeros ) ]

    return(my_extrema)

def make_dataset(records, data_path, ds_config, leads_x_rec = [], data_aumentation = 1, ds_name = 'none'):


    # Recorro los archivos
#    for this_rec in records:
    
    if len(leads_x_rec) == 0:
        leads_x_rec = ['all'] * len(records) 
       
    start_beat_idx = 0
#    start_beat_idx = ((np.array(records) == 'stdb/315').nonzero())[0][0]
#    start_beat_idx = ((np.array(records) == 'ltafdb/20').nonzero())[0][0]
#        start_beat_idx = ((np.array(records) == 'edb/e0204').nonzero())[0][0]
#        start_beat_idx = ((np.array(records) == 'sddb/49').nonzero())[0][0]
    
    all_signals = []
    all_extrema = []
    ds_part = 1
    cant_total_samples = []
    parts_samples = 0
    parts_recordings = 0
    ds_parts_fn = []

    default_lead_order = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    target_lead_names =  ['II', 'V1']

    [_, target_lead_idx, _] = np.intersect1d(default_lead_order, target_lead_names,  assume_unique=True, return_indices=True)


    wt_filters = qs_filter_design( scales = np.arange(4,5), fs = ds_config['target_fs'] )

#    for this_rec in records:
    for ii in np.arange(start_beat_idx, len(records)):

        this_rec = records[ii]
        
        print ( str(my_int(ii / len(records) * 100)) + '% Procesando:' + this_rec)
        
        data, ecg_header = wf.rdsamp(os.path.join(data_path, this_rec) )

        
        [this_lead_names, this_lead_idx, _] = np.intersect1d(ecg_header['sig_name'], default_lead_order, assume_unique=True, return_indices=True)
        
        if len(this_lead_idx) > 0 and len(this_lead_idx) == len(default_lead_order):
            
            # reorder leads
            data = data[:, this_lead_idx]
            ecg_header['n_sig'] = len(this_lead_idx)
            ecg_header['sig_name'] = this_lead_names
        
        else:
            # not enough leads found !
            print('{:s}: No encontramos los leads necesarios:\n')
            [ print('  + {:s}\n'.format(this_leads)) for this_leads in ecg_header['sig_name'] ]
            

        pq_ratio = ds_config['target_fs']/ecg_header['fs']
        resample_frac = Fraction( pq_ratio ).limit_denominator(20)
        #recalculo el ratio real
        pq_ratio = resample_frac.numerator/ resample_frac.denominator
        data = sig.resample_poly(data, resample_frac.numerator, resample_frac.denominator )
        ecg_header['sig_len'] = data.shape[0]
            
        data = median_baseline_removal( data, fs = ds_config['target_fs'])
# x_st = 200000; x_off = 10000; plt.plot(data[x_st:x_st+x_off,:]); plt.show()
        
        # WT transform
        wt_data =  np.dstack([np.roll(sig.lfilter( wt_filt, 1, data, axis = 0), -int(np.round((len(wt_filt)-1)/2)), axis=0) for wt_filt in wt_filters])

        # wt_data =  np.dstack([data, wt_data]);

        wt_data = wt_data / np.linalg.norm(wt_data, 2, axis=0, keepdims=True)

        # plt.plot(np.squeeze(wt_data[0:10000,2,:]))
        # plt.pause(10)

        # calclulo los extremos relativos de mi señal en base a la wt4
        rel_extrema = my_find_extrema( np.squeeze(wt_data[:,:]), this_distance = my_int(0.15*ds_config['target_fs']) )

        # plt.figure(1); plt.clf(); jj = 1; plt.plot(np.squeeze(wt_data[0:10000,jj,:])); this_peaks = peaks[jj][peaks[jj] < 10000]; plt.plot(this_peaks, wt_data[this_peaks,jj,0], 'bx' ); plt.plot(this_peaks, wt_data[this_peaks,jj,1], 'ro' );  plt.pause(10)

        this_scale = range_estimation( data, fs = ds_config['target_fs'])
        
        k_conv = 0.6*(2**15-1);
        
        data = (np.clip(np.around(data *k_conv/this_scale ), -(2**15-1), 2**15-1 )).astype('int16')
#        plt.figure(1); plt.plot(data); plt.plot([0, data.shape[0]], [k_conv, k_conv], 'r:'); plt.plot([0, data.shape[0]], [-k_conv, -k_conv], 'r:'); this_axes = plt.gca(); this_axes.set_ylim([-(2**15-1), 2**15-1]); plt.show(1);

        all_signals += [data]
        all_extrema += [rel_extrema]
        parts_samples += data.shape[0]
        parts_recordings += 1

        if sys.getsizeof(all_signals) > ds_config['dataset_max_size']:
            
            part_fn =  'ds_' + ds_name +  '_part_' + str(ds_part) + '.npy'

            ds_parts_fn += [ part_fn ]
            cant_total_samples += [parts_samples]
            cant_total_recordings += [parts_recordings]
             
            np.save( os.path.join( ds_config['dataset_path'], part_fn),  {'signals' : all_signals,
                                                                          'rel_extrema' : all_extrema,
                                                                          'lead_names'  : default_lead_order})
            ds_part += 1
            all_signals = []
            parts_samples = 0

    if ds_part > 1 :
        # last part
        
        part_fn =  'ds_' + ds_name +  '_part_' + str(ds_part) + '.npy'

        ds_parts_fn += [ part_fn ]
        cant_total_samples += [parts_samples]
        cant_total_recordings += [parts_recordings]
             
        np.save( os.path.join( ds_config['dataset_path'], part_fn),  {'signals' : all_signals, 'rel_extrema' : all_extrema, 'lead_names'  : default_lead_order , 'cant_total_samples' : cant_total_samples})
        all_signals = []
        
        aux_df = DataFrame( { 'filename': ds_parts_fn, 
                              'ds_samples': cant_total_samples,
                              'ds_recs': cant_total_recordings
                              } )
        
    else:
        
        part_fn =  'ds_' + ds_name + '.npy'
        # unique part
        np.save( os.path.join( ds_config['dataset_path'], part_fn),  {'signals' : all_signals, 'rel_extrema' : all_extrema, 'lead_names'  : default_lead_order , 'cant_total_samples' : cant_total_samples})

        aux_df = DataFrame( { 'filename': [part_fn], 
                              'ds_samples': [parts_samples],
                              'ds_recs': [parts_recordings]
                               } )
    
    aux_df.to_csv( os.path.join(ds_config['dataset_path'], ds_name + '_size.txt'), sep=',', header=False, index=False)


parser = ap.ArgumentParser(description='Script para crear datasets para los experimentos con GANs')
parser.add_argument( 'db_path', 
                     default='/home/mariano/mariano/dbs/', 
                     type=str, 
                     help='Path a la base de datos')

parser.add_argument( '--db_name', 
                     default=None, 
                     nargs='*',
                     type=str, 
                     help='Nombre de la base de datos')

parser.add_argument( '--particion', 
                     default=None, 
                     nargs='+',
                     type=float, 
                     help='Cantidad de pacientes en el set de training-val-test')

args = parser.parse_args()

db_path = args.db_path
db_name = args.db_name
# tamaño fijo del train, el resto val y test 50% each
partition = args.particion    # patients

known_db_names = ['INCART', 'E-OTH-12-0927-015', 'biosigna']

aux_df = None

if db_name is None:
    # default databases
    db_name = known_db_names
#    db_name = ['INCART', 'E-OTH-12-0927-015' , 'biosigna']
        
    # Train-val-test
    partition_mode = '3way'

else :
    
    if len(db_name) == 1 and os.path.isfile(db_name[0]):
        # El dataset lo determina un archivo de configuración externo.
        # Train-val-test
        partition_mode = '3way'
        
        aux_df = read_csv(db_name[0], header= 0, names=['rec', 'lead'])
        
    else:
        
        db_name_found = np.intersect1d(known_db_names, db_name)
        
        if len(db_name_found) == 0:
            
            print('No pude encontrar: ' + str(db_name) )
            sys.exit(1)

        db_name = db_name_found.tolist()
        
        if partition is None:
            # Esquemas para el particionado de los datos:
            # DB completa, db_name debería ser una 
            partition_mode = 'WholeDB'
        else:
            partition_mode = '3way'

    
#if not type(db_name) != 'list':
#    # force a list        
#    
#    db_name = [db_name]


cp_path = os.path.join('.', 'checkpoint')
os.makedirs(cp_path, exist_ok=True)

dataset_path = os.path.join('.', 'datasets')
os.makedirs(dataset_path, exist_ok=True)
#dataset_path = '/tmp/datasets/'

result_path = os.path.join('.', 'results')
os.makedirs(result_path, exist_ok=True)



ds_config = { 
                'width': 10, # s
                
                'target_fs':        200, # Hz

                'heartbeat_width': .09, # (s) Width of the type of heartbeat to seek for
                'distance':  .3, # (s) Minimum separation between consequtive QRS complexes
                'explore_win': 60, # (s) window to seek for possible heartbeats to calculate scales and offset

                'max_prop_3w_x_db': [0.8, 0.1, 0.1], # máxima proporción para particionar cada DB 
                
                'data_div_train': os.path.join(dataset_path, 'data_div_train.txt'),
                'data_div_val':   os.path.join(dataset_path, 'data_div_val.txt'),
                'data_div_test':  os.path.join(dataset_path, 'data_div_test.txt'),
                
                'dataset_max_size':  800*1024**2, # bytes
#                'dataset_max_size':  3e35, # infinito bytes
#                'target_beats': 2000, # cantidad máxima de latidos por registro
                'target_beats': None, # cantidad máxima de latidos por registro
                    
                'dataset_path':   dataset_path,
                'results_path':   result_path,
                
                'train_filename': os.path.join(dataset_path, 'train_' + '_'.join(db_name) + '.npy'),
                'test_filename':  os.path.join(dataset_path, 'test_' + '_'.join(db_name) + '.npy'),
                'val_filename':   os.path.join(dataset_path, 'val_' + '_'.join(db_name) + '.npy')
             } 

bForce_data_div = True
#bForce_data_div = False


if partition_mode == 'WholeDB':

    bForce_data_div = True

else:
    
    # 3-Way split
    if np.sum( np.array(partition) ) == 1 :
        # proportions
        tgt_train_size = partition[0] # patients
        if( len(partition) > 1 ) :
            tgt_val_size = partition[1] # patients
            if( len(partition) > 2 ) :
                tgt_test_size = partition[2] # patients
            else:
                tgt_test_size = (1-tgt_train_size-tgt_val_size) # patients
        else :
            tgt_test_size = (1-tgt_train_size)/2 # patients
            tgt_val_size = (1-tgt_train_size)/2 # patients
        
    else:
        
        # absolute values
        tgt_train_size = partition[0] # patients
        
        if( len(partition) > 1 ) :
            tgt_val_size = partition[1] # patients
            if( len(partition) > 2 ) :
                tgt_test_size = partition[2] # patients
            else:
                tgt_test_size = 0 # patients
        else :
            tgt_val_size = 0 # patients
            tgt_test_size = 0 # patients
        

#if  not os.path.isfile( ds_config['train_filename'] ) or bRedo_ds:

if bForce_data_div or not os.path.isfile( ds_config['data_div_train'] ):

    if aux_df is None:
        # reviso las db
        record_names, patient_list, size_db = get_records(db_path, db_name)
        leads_x_rec = ['all'] * len(record_names)
    else:
        # los registros se imponen
        aux_df = aux_df.sort_values(by='rec')
        record_names_pedidos = aux_df['rec'].values.tolist()
        leads_x_rec = aux_df['lead'].values.tolist()
        [databases, size_db ]= np.unique([ re.split(r'\/', this_rec)[0] for this_rec in record_names_pedidos ] , return_counts = True)
        
        record_names_existentes, _, _ = get_records(db_path, databases )
        record_names_faltantes = np.setdiff1d(record_names_pedidos, record_names_existentes, assume_unique=True)
        
        if len(record_names_faltantes) > 0 :
            
            [print('falta registro: {:s} \n'.format(this_rec)) for this_rec in record_names_faltantes]
            sys.exit(1)
            
        db_name = databases.tolist()
        record_names = record_names_pedidos
        patient_list = np.arange(len(record_names))
        
        
        
    # debug
    #record_names = record_names[0:9]

    patient_indexes = np.unique(patient_list)
    cant_patients = len(patient_indexes)
#    record_names = np.unique(record_names)
    cant_records = len(record_names)


    print('\n')
    aux_str = 'Bases de datos analizadas:'
    print( aux_str )
    print( '#' * len(aux_str) )
    [ print('{:s} : {:d} pacientes'.format(this_db, this_size)) for (this_db, this_size) in zip(db_name, size_db)]
    
    print('\n\n')
    aux_str = 'TOTAL: {:d} pacientes y {:d} registros.'.format(cant_patients, cant_records)
    print( '#' * len(aux_str) )
    print( aux_str )
    print( '#' * len(aux_str) )
    
    if partition_mode == 'WholeDB':
    
        db_start = np.hstack([ 0, np.cumsum(size_db[:-1]) ])
        db_end = db_start + size_db
        db_idx = np.hstack([np.repeat(ii, size_db[ii]) for ii in range(len(size_db))])
        
        
        for jj in range(len(db_name)):
            
            aux_idx = (db_idx == jj).nonzero()
            train_patients = np.sort(np.random.choice(patient_indexes[aux_idx], size_db[jj], replace=False ))
            
            aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in train_patients]).flatten()
            this_db_recs = [record_names[my_int(ii)] for ii in aux_idx]
    
            np.savetxt( os.path.join(ds_config['dataset_path'], db_name[jj] + '_recs.txt') , this_db_recs, '%s')

            print( 'Construyendo dataset ' + db_name[jj] )
            print( '#######################################' )
        
            # Armo el set de entrenamiento, aumentando para que contemple desplazamientos temporales
            signals, labels, ds_parts = make_dataset(this_db_recs, db_path, ds_config, ds_name = db_name[jj], data_aumentation = 1 )
            
        
    elif partition_mode == '3way':
        # propocion de cada db en el dataset
        cant_pacientes = np.sum(size_db)
        
        if tgt_train_size <= 1 and tgt_train_size >= 0 :
            tgt_train_size = my_int(cant_pacientes * tgt_train_size)
        
        if tgt_val_size <= 1 and tgt_val_size >= 0 :
            tgt_val_size = my_int(cant_pacientes * tgt_val_size)
            
        if tgt_test_size <= 1 and tgt_test_size >= 0 :
            tgt_test_size = my_int(cant_pacientes * tgt_test_size)
            
        print('\n')
        aux_str = 'Construyendo datasets ' ' Train: {:3.0f} - Val: {:3.0f} - Test: {:3.0f}'.format(tgt_train_size, tgt_val_size, tgt_test_size)            
        print( aux_str )
        print( '#' * len(aux_str) )
            
        prop_db = size_db / cant_pacientes
        
#        # particionamiento de 3 vías 
#        # train      80%
#        # validation 10%
#        # eval       10%
#        tgt_train_size = my_int(cant_patients * 0.8)
        
        
        # proporciones del corpus completo
#        tgt_db_parts_size = tgt_train_size * prop_db
#        tgt_db_parts_size = [ my_int(ii) for ii in tgt_db_parts_size]

        tgt_train_db_parts_size = tgt_train_size * np.ones(len(prop_db)) / len(prop_db)
        tgt_train_db_parts_size = [ my_ceil(ii) for ii in tgt_train_db_parts_size]
        
        tgt_val_db_parts_size = tgt_val_size * np.ones(len(prop_db)) / len(prop_db)
        tgt_val_db_parts_size = [ my_ceil(ii) for ii in tgt_val_db_parts_size]
        
        tgt_test_db_parts_size = tgt_test_size * np.ones(len(prop_db)) / len(prop_db)
        tgt_test_db_parts_size = [ my_ceil(ii) for ii in tgt_test_db_parts_size]
        
        db_start = np.hstack([ 0, np.cumsum(size_db[:-1]) ])
        db_end = db_start + size_db
        db_idx = np.hstack([np.repeat(ii, size_db[ii]) for ii in range(len(size_db))])
        
        train_recs = []
        val_recs = []
        test_recs = []
        
        train_leads_x_rec = []
        val_leads_x_rec = []
        test_leads_x_rec = []
        
        max_prop_3w_x_db = ds_config['max_prop_3w_x_db']
        
        for jj in range(len(db_name)):
            
            aux_idx = (db_idx == jj).nonzero()
            np.random.randint(len(aux_idx))
            train_patients = np.sort(np.random.choice(patient_indexes[aux_idx], np.min( [my_int(size_db[jj]*max_prop_3w_x_db[0]),  tgt_train_db_parts_size[jj] ]), replace=False ))
            test_patients = np.sort(np.setdiff1d(patient_indexes[aux_idx], train_patients, assume_unique=True))
            # test y val serán la misma cantidad de pacientes
            val_patients = np.sort(np.random.choice(test_patients, np.min( [len(test_patients)-1, my_int(size_db[jj]*max_prop_3w_x_db[1]), tgt_val_db_parts_size[jj] ]), replace=False ))
            test_patients = np.setdiff1d(test_patients, val_patients, assume_unique=True)

            test_patients = np.sort(np.random.choice(test_patients, np.min( [len(test_patients), my_int(size_db[jj]*max_prop_3w_x_db[2]), tgt_test_db_parts_size[jj] ]), replace=False ))
            
            if len(train_patients) > 0 :
                aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in train_patients]).flatten()
                aux_val = [record_names[my_int(ii)] for ii in aux_idx]
                train_recs += aux_val
                
                aux_val = [leads_x_rec[my_int(ii)] for ii in aux_idx]
                train_leads_x_rec += aux_val
    
            if len(val_patients) > 0 :
                aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in val_patients]).flatten()
                aux_val = [record_names[my_int(ii)] for ii in aux_idx]
                val_recs += aux_val

                aux_val = [leads_x_rec[my_int(ii)] for ii in aux_idx]
                val_leads_x_rec += aux_val

            if len(test_patients) > 0 :
                aux_idx = np.hstack([ (patient_list==pat_idx).nonzero() for pat_idx in test_patients]).flatten()
                aux_val = [record_names[my_int(ii)] for ii in aux_idx]
                test_recs += aux_val
                
                aux_val = [leads_x_rec[my_int(ii)] for ii in aux_idx]
                test_leads_x_rec += aux_val
        
    #    # particionamiento de 3 vías 
    #    # train      80%
    #    # validation 20%
    #    # eval       20%
    #    train_recs = np.random.choice(record_names, int(cant_records * 0.8), replace=False )
    #    test_recs = np.setdiff1d(record_names, train_recs, assume_unique=True)
    #    val_recs = np.random.choice(train_recs, int(cant_records * 0.2), replace=False )
    #    train_recs = np.setdiff1d(train_recs, val_recs, assume_unique=True)
        
    #    data_path = os.path.join(db_path, db_name)
    
        np.savetxt(ds_config['data_div_train'], train_recs, '%s')
        np.savetxt(ds_config['data_div_val'], val_recs, '%s')
        np.savetxt(ds_config['data_div_test'], test_recs, '%s')
 
else:
        
    train_recs = np.loadtxt(ds_config['data_div_train'], dtype=str )
    val_recs = np.loadtxt(ds_config['data_div_val'], dtype=str)
    test_recs = np.loadtxt(ds_config['data_div_test'], dtype=str)


if partition_mode == '3way':
    
    if len(train_recs) > 0 :
    
        print('\n')
        print( 'Construyendo el train' )
        print( '#####################' )
    
        # Armo el set de entrenamiento, aumentando para que contemple desplazamientos temporales
        make_dataset(train_recs, db_path, ds_config, leads_x_rec = train_leads_x_rec, ds_name = 'train', data_aumentation = 1 )

    if len(val_recs) > 0 :
    
        print('\n')
        print( 'Construyendo el val' )
        print( '###################' )
        # Armo el set de validacion
        make_dataset(val_recs, db_path, ds_config, leads_x_rec = val_leads_x_rec, ds_name = 'val')

    if len(test_recs) > 0 :

        print('\n')
        print( 'Construyendo el test' )
        print( '####################' )
        # Armo el set de testeo
        make_dataset(test_recs, db_path, ds_config, leads_x_rec = test_leads_x_rec, ds_name = 'test')

        
