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
from scipy.signal import medfilt
import time
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
from statsmodels.robust.scale import mad
import re


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

def get_beats(annotation):

    no_beats = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"', '@']
    
    not_beats_mk = np.isin(annotation.symbol, no_beats, assume_unique=True)
    beats_mk = np.logical_and( np.ones_like(not_beats_mk), np.logical_not(not_beats_mk) )

    # Me quedo con las posiciones
    beats = annotation.sample[beats_mk]

    return beats

def gen_interest_ranges( start_end, references, width):
    
    no_qRS_ranges, qRS_ranges = [], []
    
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
        
        qRS_ranges = []
        no_qRS_ranges = []
        
#        while st_idx < nrefs:
#
#            if this_end - width > this_start:
#                
#                no_qRS_ranges += [ ( this_start , this_end )  ]
#
#                this_qstart = references[st_idx] - half_win
#                this_qend = this_qstart + width
#                qRS_ranges += [ ( this_qstart, this_qend )  ]
#                
#                break
#            
#            else:
#
# lo descarto porque no me interesa que pudiera ser más corto que  width               
#                this_qstart = np.max( (0, references[st_idx] - half_win) )
#                this_qend = this_qstart + width
#                qRS_ranges += [ ( this_qstart, this_qend )  ]
                
#                this_start = this_end + width + 2
#                this_end = references[st_idx+1] - half_win - 1
#                st_idx += 1

# probar esto
#        qRS_ranges = [ (this_ref - half_win) for this_ref in references ]
#        qRS_ranges = [ [this_ref, this_ref + width ] for this_ref in qRS_ranges ]
#        no_qRS_ranges = np.transpose(np.hstack(qRS_ranges))
#        no_qRS_ranges = np.transpose(np.hstack([ [ no_qRS_ranges[ii,0], no_qRS_ranges[ii-1,1] ] for ii in range(len(qRS_ranges)) ]))
#        no_qRS_ranges = no_qRS_ranges[ np.diff(no_qRS_ranges, axis=1) >= width,:]
        
        for ii in np.arange(st_idx, nrefs):
            
            this_start = this_end + width + 2
            this_end = references[ii] - half_win - 1
            
            if this_end - width > this_start:
                
                no_qRS_ranges += [ ( this_start , this_end )  ]

            this_qstart = references[ii] - half_win
# con numpy clipea automáticamente
#            this_qend = np.min( ( start_end[1], this_qstart + width) )
            this_qend = this_qstart + width
            qRS_ranges += [ ( this_qstart, this_qend )  ]

#        # última referencia
#        this_start = this_end + width + 2
#        this_end = references[-1] - half_win - 1
#        
#        if this_end - width > this_start:
#            
#            no_qRS_ranges += [ ( this_start , this_end )  ]
#    
#        # posible último tramo
#        this_start = this_end + width + 2
#        this_end = start_end[1]
#        
#        if this_end - width > this_start:
#            
#            no_qRS_ranges += [ ( this_start , this_end )  ]
        
    
    return no_qRS_ranges, qRS_ranges

def my_int(x):
    
    return int(np.round(x))

def my_ceil(x):
    
    return int(np.ceil(x))

def make_dataset(records, data_path, ds_config, leads_x_rec = [], data_aumentation = 1, ds_name = 'none'):

    signals, labels = [], []
    
    
    nQRS_QRS_ratio = []
    cant_latidos_total = 0

    # Recorro los archivos
#    for this_rec in records:
    
    if len(leads_x_rec) == 0:
        leads_x_rec = ['all'] * len(records) 
       
    start_beat_idx = 0
#    start_beat_idx = ((np.array(records) == 'stdb/315').nonzero())[0][0]
#    start_beat_idx = ((np.array(records) == 'ltafdb/20').nonzero())[0][0]
#        start_beat_idx = ((np.array(records) == 'edb/e0204').nonzero())[0][0]
#        start_beat_idx = ((np.array(records) == 'sddb/49').nonzero())[0][0]

    tgt_ratio = np.nan


    min_cant_latidos = 34e10;
        
        
    if np.isnan(ds_config['tgt_ratio']):
    
        for ii in np.arange(start_beat_idx, len(records)):
            
            this_rec = records[ii]
            print ('Procesando:' + this_rec)
            data, field = wf.rdsamp(os.path.join(data_path, this_rec) )
            annotations = wf.rdann(os.path.join(data_path, this_rec), 'atr')
    
            # filtro los latidos 
            beats = get_beats(annotations)
    
            this_cant_latidos = len(beats)
            
            min_cant_latidos = np.min( [min_cant_latidos, this_cant_latidos ] )
    
            cant_latidos_total += this_cant_latidos
            
            w_in_samp = my_int( ds_config['width'] * field['fs'])
            hw_in_samp = my_int( ds_config['width'] * field['fs'] / 2)
            tol_in_samp = my_int( ds_config['heartbeat_tolerance'] * field['fs'])
            samp_around_beats = w_in_samp * len(beats)
            # acumulo los ratios de QRS sobre el total de muestras para mantener ese ratio en el train ds
            this_ratio = (field['sig_len'] - samp_around_beats)/samp_around_beats
            nQRS_QRS_ratio.append(this_ratio)
            
        # target proportion ratio 
        tgt_ratio = np.median(nQRS_QRS_ratio)
                
    else:
         tgt_ratio = ds_config['tgt_ratio']
            
         
 
    if not ds_config['target_beats'] is None:
        
        print('*********************************************')
        print ('Construyendo para ' + str(ds_config['target_beats']) + ' latidos por paciente.')
        print ('El ratio no_latido/latido es: {:3.3f}'.format(tgt_ratio) )
        print('*********************************************')

        if ds_config['target_beats'] > min_cant_latidos:
            print('*********************************************')
            print ('OJALDRE!  Hay registros con menos latidos: ' + str(min_cant_latidos) + ' latidos')
            print('*********************************************')
        
    
    start_beat_idx = 0
    
    all_signals = []
    ds_part = 1
    cant_total_samples = 0
    ds_parts_fn = []
    ds_parts_size = []
    ds_parts_features = []

    w_in_samp = my_int( ds_config['width'] * ds_config['target_fs'])
    hw_in_samp = my_int( ds_config['width'] * ds_config['target_fs'] / 2)
    tol_in_samp = my_int( ds_config['heartbeat_tolerance'] * ds_config['target_fs'])

#    for this_rec in records:
    for ii in np.arange(start_beat_idx, len(records)):

        this_rec = records[ii]
        this_leads_idx = leads_x_rec[ii]
        
        print ( str(my_int(ii / len(records) * 100)) + '% Procesando:' + this_rec)
        
        data, field = wf.rdsamp(os.path.join(data_path, this_rec) )
        annotations = wf.rdann(os.path.join(data_path, this_rec), 'atr')

        if this_leads_idx != 'all' :
            
            [_, this_leads_idx, _] = np.intersect1d(field['sig_name'], this_leads_idx.strip(), assume_unique=True, return_indices=True)
            
            if len(this_leads_idx) > 0 :
                data = data[:, this_leads_idx]
                field['n_sig'] = len(this_leads_idx)

        pq_ratio = ds_config['target_fs']/field['fs']
        resample_frac = Fraction( pq_ratio ).limit_denominator(20)
        #recalculo el ratio real
        pq_ratio = resample_frac.numerator/ resample_frac.denominator
        data = sig.resample_poly(data, resample_frac.numerator, resample_frac.denominator )
        
        # filtro los latidos 
        beats = get_beats(annotations)
        # resample references
        beats = np.round(beats * pq_ratio)
        
        this_cant_beats = len(beats)

        # genero las referencias temporales para generar los vectores de entrada
        no_qRS_ranges, qRS_ranges = gen_interest_ranges( start_end=[0, np.round( field['sig_len'] * pq_ratio)  ], references = beats, width = w_in_samp )

        if not ds_config['target_beats'] is None and ds_config['target_beats'] <= this_cant_beats :
            this_beats_idx = np.sort(np.random.choice( np.arange(len(qRS_ranges)), ds_config['target_beats'], replace=False ))
            qRS_ranges = np.vstack(qRS_ranges)
            qRS_ranges = qRS_ranges[this_beats_idx,:]
            qRS_ranges = qRS_ranges.tolist()


        this_cant_no_beats = my_ceil( len(qRS_ranges) * tgt_ratio )
        
        if this_cant_no_beats < len(no_qRS_ranges):
            # solo me quedo con un subset aleatorio
            this_beats_idx = np.sort(np.random.choice( np.arange(len(no_qRS_ranges)), this_cant_no_beats, replace=False ))
            
            no_qRS_ranges = np.vstack(no_qRS_ranges)
            no_qRS_ranges = no_qRS_ranges[this_beats_idx,:]
            no_qRS_ranges = no_qRS_ranges.tolist()
            
        else:
            # aumento la cantidad de segmentos no_QRS de acuerdo al ratio deseado 
            segments_repeat = my_int(np.abs( len(no_qRS_ranges) - np.ceil( len(qRS_ranges) * tgt_ratio ) ))
            no_qRS_ranges += [ no_qRS_ranges[np.random.randint(len(no_qRS_ranges))] for _ in range(segments_repeat) ]
        
        # genero los comienzos aumentados de acuerdo a data_aumentation
        starts = []
        
        starts += [ (np.random.randint(this_start_end[0] - hw_in_samp, this_start_end[1] - hw_in_samp, size=data_aumentation, dtype='int')).reshape(data_aumentation,1) for this_start_end in no_qRS_ranges ]
        starts += [ (np.random.randint(this_start_end[0] - tol_in_samp, this_start_end[0] + tol_in_samp, size=data_aumentation, dtype='int')).reshape(data_aumentation,1) for this_start_end in qRS_ranges ]
#        starts += [ this_start_end[0] for this_start_end in qRS_ranges ]
        
        # 0: No QRS - 1: QRS
        this_lab = np.concatenate( ( np.zeros( ( my_int(len(no_qRS_ranges) * field['n_sig'] * data_aumentation) ,1) ), np.ones( (my_int(len(qRS_ranges) * field['n_sig'] * data_aumentation), 1) ) ), axis = 0 )
        
        # unbias and normalize
        bScaleRecording = True
#        bScaleRecording = False
        if bScaleRecording:
           
#            this_scale = mad(data, axis=0).reshape(field['n_sig'], 1 )
            
            # usando los latidos
            this_scale = (np.nanmedian(np.vstack([ np.max(np.abs(data[my_int(np.max([0, this_beat-w_in_samp])):my_int(np.min([field['sig_len'] * pq_ratio, this_beat+w_in_samp])) ,:] ), axis = 0 ) for this_beat in beats ]), axis = 0)).reshape(field['n_sig'], 1 )

            bAux = np.bitwise_or( this_scale == 0,  np.isnan(this_scale))
            if np.any(bAux):
                # avoid scaling in case 0 or NaN
                this_scale[bAux] = 1
                
                
        starts = np.vstack(starts)
        the_sigs = []
        for this_start in starts :
        
#        try:
            this_sig = np.transpose(data[my_int(this_start):my_int(this_start + w_in_samp), :]) 
                
            # unbias and normalize
            this_sig = this_sig - np.nanmedian(this_sig, axis=1, keepdims = True)
            
            if not(bScaleRecording):
                this_scale = mad(this_sig, center = 0, axis=1 ).reshape(this_sig.shape[0],1 )
                bAux = np.bitwise_or( this_scale == 0,  np.isnan(this_scale))
                if np.any(bAux):
                    # avoid scaling in case 0 or NaN
                    this_scale[bAux] = 1
                
            # add an small dither 
#            this_sig = this_sig * 1/this_scale + 1/500 * np.random.randn(this_sig.shape[0], this_sig.shape[1])
            this_sig = this_sig * 1/this_scale 

            this_sig = (np.clip(np.round(this_sig * (2**15-1) * 0.5), -(2**15-1), 2**15-1 )).astype('int16')

            the_sigs += [this_sig]
        
#        except Exception:
#            
#            a = 0
                
        
        if len(all_signals) == 0:
            all_labels = this_lab
            all_signals = np.vstack(the_sigs)
        else:
            all_labels = np.concatenate( (all_labels, this_lab) )
            all_signals = np.concatenate( (all_signals, np.vstack(the_sigs)) )

        if sys.getsizeof(all_signals) > ds_config['dataset_max_size']:
            
            part_fn =  'ds_' + ds_name +  '_part_' + str(ds_part) + '.npy'

            ds_parts_fn += [ part_fn ]
            ds_parts_size += [ all_signals.shape[0] ]
            ds_parts_features += [ all_signals.shape[1] ]
            
            cant_total_samples += all_signals.shape[0]
             
            np.save( os.path.join( ds_config['dataset_path'], part_fn),  {'signals' : all_signals,
                                                                          'labels'  : all_labels})
            ds_part += 1
            all_signals = []
            all_labels = []
            

    if ds_part > 1 :
        # last part
        
        part_fn =  'ds_' + ds_name +  '_part_' + str(ds_part) + '.npy'

        ds_parts_fn += [ part_fn ]
        ds_parts_size += [ all_signals.shape[0] ]
        ds_parts_features += [ all_signals.shape[1] ]
        
        np.save( os.path.join( ds_config['dataset_path'], part_fn),  {'signals' : all_signals,  'labels'  : all_labels , 'cant_total_samples' : all_signals.shape[0]})
        all_signals = []
        all_labels = []
        
        aux_df = DataFrame( { 'filename': ds_parts_fn, 
                              'ds_size': ds_parts_size, 
                              'ds_features': ds_parts_features, 
                              } )
        
    else:
        part_fn =  'ds_' + ds_name + '.npy'
        # unique part
        np.save( os.path.join( ds_config['dataset_path'], part_fn),  {'signals' : all_signals,  'labels'  : all_labels , 'cant_total_samples' : all_signals.shape[0] })

        aux_df = DataFrame( { 'filename': [part_fn], 
                              'ds_size': [all_signals.shape[0]],
                              'ds_features': [all_signals.shape[1]] } )
    
    aux_df.to_csv( os.path.join(ds_config['dataset_path'], ds_name + '_size.txt'), sep=',', header=False, index=False)


    return all_signals, all_labels, ds_part

parser = ap.ArgumentParser(description='Script para crear datasets')
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

known_db_names = ['stdb', 'INCART', 'mitdb' ,'ltdb' ,'E-OTH-12-0927-015' ,'ltafdb' ,'edb' ,'aha' ,'sddb' ,'svdb' ,'nsrdb' ,'ltstdb' , 'biosigna']

aux_df = None

if db_name is None or db_name == 'all' :
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
                'width': .2, # s
                'mode': 'binary', # cada ventana de *w* muestras tendrá latido/no_latido
#                'mode': 'categorical', # cada ventana de *w* muestras tendrá k categorías 
                                # 0: no_latido - 
                                # 1: QRS en la primer ventana de ancho w_1 = w/(ceil(w/150ms))
                                # ...
                                # k-1: QRS en la última ventana w_k
                'heartbeat_tolerance': .07, # s
                
                'target_fs':        250, # Hz
#                'tgt_ratio': np.nan, # nan para que lo calcule
                'tgt_ratio': 3.0, 
                
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
        signals, labels, ds_parts = make_dataset(train_recs, db_path, ds_config, leads_x_rec = train_leads_x_rec, ds_name = 'train', data_aumentation = 1 )

    if len(val_recs) > 0 :
    
        print('\n')
        print( 'Construyendo el val' )
        print( '###################' )
        # Armo el set de validacion
        signals, labels, ds_parts  = make_dataset(val_recs, db_path, ds_config, leads_x_rec = val_leads_x_rec, ds_name = 'val')

    if len(test_recs) > 0 :

        print('\n')
        print( 'Construyendo el test' )
        print( '####################' )
        # Armo el set de testeo
        signals, labels, ds_parts = make_dataset(test_recs, db_path, ds_config, leads_x_rec = test_leads_x_rec, ds_name = 'test')

        
