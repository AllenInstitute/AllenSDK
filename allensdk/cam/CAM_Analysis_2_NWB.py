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
import sys
matplotlib.use('agg')
import matplotlib.pyplot as plt
from drifting_grating import DriftingGrating
from movie_analysis import MovieAnalysis
plt.ioff()
#import Analysis.OPTools_Nikon as op
import CAM_plotting as cp
import os


class CamAnalysis2Nwb(object):
    def __init__(self, exptpath, h5path, datarate, lims_id, depth, movie_name):
        self.exptpath = exptpath
        self.h5path = h5path
        self.datarate = datarate
        self.lims_id = lims_id
        self.depth = depth
        self.movie_name = movie_name
    
    
    def save_h5_a(self, dg, nm1, nm3):                
        store = pd.HDFStore(self.h5path, mode='w')
        store['stim_table_dg'] = dg.stim_table
        store['sweep_response_dg'] = dg.sweep_response
        store['mean_sweep_response_dg'] = dg.mean_sweep_response
        
        store['sweep_response_nm1'] = nm1.sweep_response
        store['stim_table_nm1'] = nm1.stim_table
    
        store['sweep_response_nm3'] = nm3.sweep_response
        store['stim_table_nm3'] = nm3.stim_table
        #store['sweeptable'] = dg.sweeptable
        store.close()
        
        f = h5py.File(self.h5path, 'a')
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
        
    
    def save_h5_b(self, sg, nm1, ni):                
        store = pd.HDFStore(self.h5path, mode='w')
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
        f = h5py.File(self.h5path, 'a')
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
    
    
    def save_h5_c(self, lsn, nm1, nm2):                
        store = pd.HDFStore(self.h5path, mode='w')
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
        f = h5py.File(self.h5path, 'a')
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
    
    
    def stimulus_a(self, plot_flag=False, save_flag=True):
            dg = DriftingGrating(self)
            nm3 = MovieAnalysis(self, 'natural_movie_three')    
            nm1 = MovieAnalysis(self, 'natural_movie_one')        
            print "Stimulus A analyzed"
            if plot_flag:
                cp.plot_3SA(dg, nm1, nm3)
                cp.plot_Drifting_grating_Traces(dg)
    
            if save_flag:
                self.save_h5_a(dg, nm1, nm3)
    
    def stimulus_b(self, plot_flag=False, save_flag=True):
                sg = StaticGrating(self)    
                ni = NaturalImages(self)
                nm1 = MovieAnalysis(self, 'natural_movie_one')            
                print "Stimulus B analyzed"
                if plot_flag:
                    cp.plot_3SB(sg, nm1, ni)
                    cp.plot_NI_Traces(ni)
                    cp.plot_SG_Traces(sg)
                    
                if save_flag:
                    self.save_h5_b(sg, nm1, ni)
    
    def stimulus_c(self, plot_flag=False, save_flag=True):
                nm2 = MovieAnalysis(self, 'natural_movie_two')
                lsn = LocallySN(self)
                nm1 = MovieAnalysis(self, 'natural_movie_one')
                print "Stimulus C analyzed"
                
                if plot_flag:
                    cp.plot_3SC(lsn, nm1, nm2)
                    cp.plot_LSN_Traces(lsn)
    
                if save_flag:
                    self.save_h5_c(lsn, nm1, nm2)

if __name__=='__main__':
    stimulus = sys.argv[-1]
    
    if 'A' == stimulus:
        lims_id = '501836392' # A        
    elif 'B' == stimulus:
        lims_id = '501886692' # B
    elif 'C' == stimulus:
        lims_id = '501717543'  # C
    else:
        raise(Exception('please specify stimulus A, B or C'))

#    Cre = 'Cux2'
#    HVA = 'AL'
    depth= 175
    cam_directory = '/local1/cam_datasets'               
    exptpath = os.path.join(cam_directory, lims_id)
    h5path = os.path.join(exptpath, 'Data', 'Data_timf.h5')
    datarate = 30    
    movie_name = 'natural_movie_one'

    cam_analysis_2_nwb = CamAnalysis2Nwb(exptpath, h5path, datarate, lims_id, depth, movie_name)

    if 'A' == stimulus:
        cam_analysis_2_nwb.stimulus_a()
    elif 'B' == stimulus:
        cam_analysis_2_nwb.stimulus_b()
    else:
        cam_analysis_2_nwb.stimulus_c()
    
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
