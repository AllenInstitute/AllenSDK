# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:26:59 2015

@author: saskiad
"""

import matplotlib
import h5py
import pandas as pd
from allensdk.cam.static_grating import StaticGrating
from allensdk.cam.movie_analysis import LocallySN
from allensdk.cam.natural_images import NaturalImages
matplotlib.use('agg')
import matplotlib.pyplot as plt
from drifting_grating import DriftingGrating
from movie_analysis import MovieAnalysis
plt.ioff()
#import Analysis.OPTools_Nikon as op
import CAM_plotting as cp
import os

def save_h5_a(dg, nm1, nm3):                
    filename = 'Data_timf.h5'
    fullfilename = os.path.join(dg.savepath, filename)
    store = pd.HDFStore(fullfilename, mode='w')
    store['stim_table_dg'] = dg.stim_table
    store['sweep_response_dg'] = dg.sweep_response
    store['mean_sweep_response_dg'] = dg.mean_sweep_response
    
    store['sweep_response_nm1'] = nm1.sweep_response
    store['stim_table_nm1'] = nm1.stim_table

    store['sweep_response_nm3'] = nm3.sweep_response
    store['stim_table_nm3'] = nm3.stim_table
    #store['sweeptable'] = dg.sweeptable
    store.close()
    
    f = h5py.File(fullfilename, 'a')
    f.create_dataset('celltraces', data=dg.celltraces)
    #f.create_dataset('twop_frames', data=dg.twop_frames)
    f.create_dataset('acquisition_rate', data=dg.acquisition_rate)
    f.create_dataset('celltraces_dff', data=nm1.celltraces_dff)
    #dset5 = f.create_dataset('dxcm', data=sg.dxcm)  
    f.create_dataset('response_dg', data=dg.response)

    f.create_dataset('binned_cells_sp', data=nm1.binned_cells_sp)
    f.create_dataset('binned_cells_vis', data=nm1.binned_cells_vis)
    f.create_dataset('binned_dx_sp', data=nm1.binned_dx_sp)
    f.create_dataset('binned_dx_vis',data=nm1.binned_dx_vis)
    ##f.keys()
    f.close()
    

def save_h5_b(sg, nm1, ni):                
    filename = 'Data_timf.h5'
    fullfilename = os.path.join(sg.savepath, filename)
    store = pd.HDFStore(fullfilename, mode='w')
    store['stim_table_sg'] = sg.stim_table
    store['sweep_response_sg'] = sg.sweep_response
    store['mean_sweep_response_sg'] = sg.mean_sweep_response
    store['sweep_response_nm1'] = nm1.sweep_response
    store['stim_table_nm1'] = nm1.stim_table
    store['sweep_response_ni'] = ni.sweep_response
    store['stim_table_ni'] = ni.stim_table
    store['mean_sweep_response_ni'] = ni.mean_sweep_response
    #store['sweeptable'] = dg.sweeptable
    store.close()
    f = h5py.File(fullfilename, 'a')
    #dset6 = f.create_dataset('receptive_field', data=lsn.receptive_field)
    #dset = f.create_dataset('celltraces', data=sg.celltraces)
    #dset2 = f.create_dataset('twop_frames', data=sg.twop_frames)
    #dset3 = f.create_dataset('acquisition_rate', data=sg.acquisition_rate)
    f.create_dataset('celltraces_dff', data=nm1.celltraces_dff)
    f.create_dataset('dxcm', data=sg.dxcm)  
    f.create_dataset('response_sg', data=sg.response)
    f.create_dataset('response_ni', data=ni.response)
    f.create_dataset('binned_cells_sp', data=nm1.binned_cells_sp)
    f.create_dataset('binned_cells_vis', data=nm1.binned_cells_vis)
    f.create_dataset('binned_dx_sp', data=nm1.binned_dx_sp)
    f.create_dataset('binned_dx_vis',data=nm1.binned_dx_vis)
    ##f.keys()
    f.close()


def save_h5_c(lsn, nm1, nm2):                
    filename = 'Data_timf.h5'
    fullfilename = os.path.join(lsn.savepath, filename)
    store = pd.HDFStore(fullfilename, mode='w')
    store['stim_table_lsn'] = lsn.stim_table
    store['sweep_response'] = lsn.sweep_response
    store['mean_sweep_response'] = lsn.mean_sweep_response
    
    store['sweep_response_nm1'] = nm1.sweep_response
    #store['stim_table_nm1'] = nm1.stim_table # weren't in orig Data.h5

    store['sweep_response_nm2'] = nm2.sweep_response
    #store['stim_table_nm2'] = nm2.stim_table  # weren't in orig Data.h5
    
    store['sweep_response_lsn'] = lsn.sweep_response
    store['stim_table_lsn'] = lsn.stim_table
    store['mean_sweep_response_lsn'] = lsn.mean_sweep_response
    #store['sweeptable'] = dg.sweeptable
    store.close()
    f = h5py.File(fullfilename, 'a')
    f.create_dataset('receptive_field_lsn', data=lsn.receptive_field)
    f.create_dataset('celltraces', data=lsn.celltraces)
    #f.create_dataset('twop_frames', data=lsn.twop_frames)
    f.create_dataset('acquisition_rate', data=lsn.acquisition_rate)
    f.create_dataset('celltraces_dff', data=nm1.celltraces_dff)
    #f.create_dataset('dxcm', data=lsn.dxcm)  
    #f.create_dataset('response_lsn', data=lsn.response)
    f.create_dataset('binned_dx_sp', data=nm1.binned_dx_sp)
    f.create_dataset('binned_dx_vis',data=nm1.binned_dx_vis)    
    f.create_dataset('binned_cells_sp', data=nm1.binned_cells_sp)
    f.create_dataset('binned_cells_vis', data=nm1.binned_cells_vis)
    ##f.keys()
    f.close()


def stimulus_a(exptpath, h5path, datarate, LIMSID, depth, movie_name,
               plot_flag=False, save_flag=True):
        dg = DriftingGrating(exptpath, h5path, datarate, LIMSID, depth, movie_name)
        movie_name = 'natural_movie_three'
        nm3 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, depth, movie_name)    
        movie_name = 'natural_movie_one'
        nm1 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, depth, movie_name)        
        print "Stimulua A analyzed"
        if plot_flag:
            cp.plot_3SA(dg, nm1, nm3)
            cp.plot_Drifting_grating_Traces(dg)

        if save_flag:
            save_h5_a(dg, nm1, nm3)

def stimulus_b(exptpath, h5path, datarate, LIMSID, depth, movie_name,
               plot_flag=False, save_flag=True):
            sg = StaticGrating(exptpath, h5path, datarate, LIMSID, depth, movie_name)    
            ni = NaturalImages(exptpath, h5path, datarate, LIMSID, depth, movie_name)
            movie_name = 'natural_movie_one'
            nm1 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, depth, movie_name)            
            print "Stimulus B analyzed"
            if plot_flag:
                cp.plot_3SB(sg, nm1, ni)
                cp.plot_NI_Traces(ni)
                cp.plot_SG_Traces(sg)
                
            if save_flag:
                save_h5_b(sg, nm1, ni)

def stimulus_c(exptpath, h5path, datarate, LIMSID, depth, movie_name,
               plot_flag=False, save_flag=True):
            movie_name = 'natural_movie_two'    
            nm2 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, depth, movie_name)
            lsn = LocallySN(exptpath, h5path, datarate, LIMSID, depth, movie_name)
            movie_name = 'natural_movie_one'
            nm1 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, depth, movie_name)
            print "Stimulus C analyzed"
            
            if plot_flag:
                cp.plot_3SC(lsn, nm1, nm2)
                cp.plot_LSN_Traces(lsn)

            if save_flag:
                save_h5_c(lsn, nm1, nm2)

if __name__=='__main__':
    LIMSID = '501836392' # A
    #LIMSID = '501886692' # B
    #LIMSID = '501717543'  # C
#    Cre = 'Cux2'
#    HVA = 'AL'
    depth= 175
    cam_directory = '/local1/cam_datasets'               
    exptpath = os.path.join(cam_directory, LIMSID)
    h5path = None
    datarate = 30    
    movie_name = 'natural_movie_one'
    stimulus_args = (exptpath, h5path, datarate, LIMSID, depth, movie_name)

    stimulus_a(*stimulus_args)
#    stimulus_b(*stimulus_args)
#    stimulus_c(*stimulus_args)
    
    
#    print "Number of cells:", str(lsn.numbercells)
#    cp.plot_3SB(sg, nm1, ni)
#    cp.plot_NI_Traces(ni)
#    cp.plot_SG_Traces(sg)
    #cp.plot_3SC(lsn, nm1, nm2)
    #cp.plot_LSN_Traces(lsn)
    
#    lsn.plotAll()
#    dg.saveh5()

#    dg.plotTraces()   
#    dg.ExperimentSummary()
#    dg.plotRunning()    
    
    
#
##
####            
