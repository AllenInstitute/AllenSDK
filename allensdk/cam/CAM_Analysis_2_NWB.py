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


class CamAnalysis2Nwb(object):
    def __init__(self, nwb_path, savepath, datarate, lims_id, depth):
        self.nwb_path = nwb_path
        self.savepath = save_path
        self.datarate = datarate
        self.lims_id = lims_id
        self.depth = depth


    def save_analysis_hdf5(self, *tables):
        store = pd.HDFStore(self.nwb_path, mode='a')

        for k,v in tables:
            store.put('analysis/%s' % (k), v)

        store.close()
        

    def save_analysis_datasets(self, *datasets):    
        f = h5py.File(self.nwb_path, 'a')
        
        for k,v in datasets:
            if k in f['analysis']:
                del f['analysis'][k]
            f.create_dataset('analysis/%s' % k, data=v)
        
        ##f.keys()
        f.close()
    
    def save_h5_a(self, dg, nm1, nm3):                
        self.save_analysis_hdf5(
            ('stim_table_dg', dg.stim_table),
            ('sweep_response_dg', dg.sweep_response),
            ('mean_sweep_response_dg', dg.mean_sweep_response),
            ('peak_dg', dg.peak),        
            ('sweep_response_nm1', nm1.sweep_response),
            ('stim_table_nm1', nm1.stim_table),
            ('peak_nm1', nm1.peak),
            ('sweep_response_nm3', nm3.sweep_response),
            ('stim_table_nm3', nm3.stim_table),
            ('peak_nm3', nm3.peak))
        
        self.save_analysis_datasets(
            ('celltraces', dg.celltraces),
            ('acquisition_rate', dg.acquisition_rate),
            ('celltraces_dff', nm1.celltraces_dff),
            ('response_dg', dg.response),
            ('binned_cells_sp', nm1.binned_cells_sp),
            ('binned_cells_vis', nm1.binned_cells_vis),
            ('binned_dx_sp', nm1.binned_dx_sp),
            ('binned_dx_vis', nm1.binned_dx_vis))
    
        
    def save_h5_b(self, sg, nm1, ni):                
        self.save_analysis_hdf5(
            ('stim_table_sg', sg.stim_table),
            ('sweep_response_sg', sg.sweep_response),
            ('mean_sweep_response_sg', sg.mean_sweep_response),
            ('sweep_response_nm1', nm1.sweep_response),
            ('stim_table_nm1', nm1.stim_table),
            ('sweep_response_ni', ni.sweep_response),
            ('stim_table_ni', ni.stim_table),
            ('mean_sweep_response_ni', ni.mean_sweep_response),
            ('peak_sg', sg.peak),
            ('peak_ni', ni.peak),
            ('peak_nm1', nm1.peak))

        self.save_analysis_datasets(
            ('celltraces_dff', nm1.celltraces_dff),
            ('dxcm', sg.dxcm),  
            ('response_sg', sg.response),
            ('response_ni', ni.response),
            ('binned_cells_sp', nm1.binned_cells_sp),
            ('binned_cells_vis', nm1.binned_cells_vis),
            ('binned_dx_sp', nm1.binned_dx_sp),
            ('binned_dx_vis', nm1.binned_dx_vis))
    
    
    def save_h5_c(self, lsn, nm1, nm2):                
        self.save_analysis_hdf5(
            ('stim_table_lsn', lsn.stim_table),
            ('sweep_response', lsn.sweep_response),
            ('mean_sweep_response', lsn.mean_sweep_response),
            ('sweep_response_nm1', nm1.sweep_response),
            ('peak_nm1', nm1.peak),
            ('sweep_response_nm2', nm2.sweep_response),
            ('peak_nm2', nm2.peak),
            ('sweep_response_lsn', lsn.sweep_response),
            ('stim_table_lsn', lsn.stim_table),
            ('mean_sweep_response_lsn', lsn.mean_sweep_response))  
        
        self.save_analysis_datasets(
            ('receptive_field_lsn', lsn.receptive_field),
            ('celltraces', lsn.celltraces),
            ('acquisition_rate', lsn.acquisition_rate),
            ('celltraces_dff', nm1.celltraces_dff),
            ('binned_dx_sp', nm1.binned_dx_sp),
            ('binned_dx_vis', nm1.binned_dx_vis),    
            ('binned_cells_sp', nm1.binned_cells_sp),
            ('binned_cells_vis', nm1.binned_cells_vis))
    
    
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
    try:
        (stimulus, nwb_path, save_path, lims_id) = sys.argv[-4:]  
        # A /local1/cam_datasets/501836392/501836392.nwb /local1/cam_datasets/501836392/Data 501836392        
        # B /local1/cam_datasets/501886692/501886692.nwb /local1/cam_datasets/501886692/Data 501886692
        # C /local1/cam_datasets/501717543/501717543.nwb /local1/cam_datasets/501717543/Data 501717543
    except:
        raise(Exception('please specify stimulus A, B or C, cam_directory, lims_id'))

#    Cre = 'Cux2'
#    HVA = 'AL'
    depth= 175
    datarate = 30    

    cam_analysis_2_nwb = CamAnalysis2Nwb(nwb_path, save_path, datarate, lims_id, depth)

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
