# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:26:59 2015

@author: saskiad
"""

#import matplotlib
from allensdk.cam.static_grating import StaticGrating
from allensdk.cam.movie_analysis import LocallySN
from allensdk.cam.natural_images import NaturalImages
from allensdk.cam.CAM_NWB import CamNwbDataSet
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
from drifting_grating import DriftingGrating
from movie_analysis import MovieAnalysis
#plt.ioff()
import CAM_plotting as cp
import pandas as pd 
import os

class CamAnalysis2Nwb(object):
    def __init__(self, nwb_path, save_path, depth):
        self.nwb = CamNwbDataSet(nwb_path)                        
        self.save_path = save_path
        self.depth = depth
    
    def save_h5_a(self, dg, nm1, nm3, peak):
        nwb = CamNwbDataSet(self.save_path)
        nwb.save_analysis_dataframes(
            ('stim_table_dg', dg.stim_table),
            ('sweep_response_dg', dg.sweep_response),
            ('mean_sweep_response_dg', dg.mean_sweep_response),
            ('peak', peak),        
            ('sweep_response_nm1', nm1.sweep_response),
            ('stim_table_nm1', nm1.stim_table),
            ('sweep_response_nm3', nm3.sweep_response))
        
        nwb.save_analysis_arrays(
            ('celltraces_dff', nm1.celltraces_dff),
            ('response_dg', dg.response),
            ('binned_cells_sp', nm1.binned_cells_sp),
            ('binned_cells_vis', nm1.binned_cells_vis),
            ('binned_dx_sp', nm1.binned_dx_sp),
            ('binned_dx_vis', nm1.binned_dx_vis))
    
        
    def save_h5_b(self, sg, nm1, ni, peak): 
        nwb = CamNwbDataSet(self.save_path)
        nwb.save_analysis_dataframes(
            ('stim_table_sg', sg.stim_table),
            ('sweep_response_sg', sg.sweep_response),
            ('mean_sweep_response_sg', sg.mean_sweep_response),
            ('sweep_response_nm1', nm1.sweep_response),
            ('stim_table_nm1', nm1.stim_table),
            ('sweep_response_ni', ni.sweep_response),
            ('stim_table_ni', ni.stim_table),
            ('mean_sweep_response_ni', ni.mean_sweep_response),
            ('peak', peak))

        nwb.save_analysis_arrays(
            ('celltraces_dff', nm1.celltraces_dff),
            ('response_sg', sg.response),
            ('response_ni', ni.response),
            ('binned_cells_sp', nm1.binned_cells_sp),
            ('binned_cells_vis', nm1.binned_cells_vis),
            ('binned_dx_sp', nm1.binned_dx_sp),
            ('binned_dx_vis', nm1.binned_dx_vis))
    
    
    def save_h5_c(self, lsn, nm1, nm2, peak):                
        nwb = CamNwbDataSet(self.save_path)
        nwb.save_analysis_dataframes(
            ('stim_table_lsn', lsn.stim_table),
            ('sweep_response_nm1', nm1.sweep_response),
            ('peak', peak),
            ('sweep_response_nm2', nm2.sweep_response),
            ('sweep_response_lsn', lsn.sweep_response),
            ('mean_sweep_response_lsn', lsn.mean_sweep_response))  
        
        nwb.save_analysis_arrays(
            ('receptive_field_lsn', lsn.receptive_field),
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
        peak = pd.concat([nm1.peak_run, dg.peak, nm1.peak, nm3.peak], axis=1)
        if plot_flag:
            cp.plot_3SA(dg, nm1, nm3)
            cp.plot_Drifting_grating_Traces(dg)
    
        if save_flag:
            self.save_h5_a(dg, nm1, nm3, peak)
    
    def stimulus_b(self, plot_flag=False, save_flag=True):
                sg = StaticGrating(self)    
                ni = NaturalImages(self)
                nm1 = MovieAnalysis(self, 'natural_movie_one')            
                print "Stimulus B analyzed"
                peak = pd.concat([nm1.peak_run, sg.peak, ni.peak, nm1.peak], axis=1)
                
                if plot_flag:
                    cp.plot_3SB(sg, nm1, ni)
                    cp.plot_NI_Traces(ni)
                    cp.plot_SG_Traces(sg)
                    
                if save_flag:
                    self.save_h5_b(sg, nm1, ni, peak)
    
    def stimulus_c(self, plot_flag=False, save_flag=True):
                nm2 = MovieAnalysis(self, 'natural_movie_two')
                lsn = LocallySN(self)
                nm1 = MovieAnalysis(self, 'natural_movie_one')
                print "Stimulus C analyzed"
                peak = pd.concat([nm1.peak_run, nm1.peak, nm2.peak], axis=1)
                
                if plot_flag:
                    cp.plot_3SC(lsn, nm1, nm2)
                    cp.plot_LSN_Traces(lsn)
    
                if save_flag:
                    self.save_h5_c(lsn, nm1, nm2, peak)
                    
def main(stimulus, nwb_path, save_path, depth):   

    cam_analysis_2_nwb = CamAnalysis2Nwb(nwb_path, save_path, depth)

    if 'A' == stimulus:
        cam_analysis_2_nwb.stimulus_a(plot_flag=False)
    elif 'B' == stimulus:
        cam_analysis_2_nwb.stimulus_b(plot_flag=False)
    else:
        cam_analysis_2_nwb.stimulus_c(plot_flag=False)
    
   

if __name__=='__main__':
    lims_id = 502115959
    depth = 175
    stimulus = 'A'
    save_dir = r'/Users/saskiad/Documents/Data/ophysdev/'
    nwb_path = os.path.join(save_dir, "%d/%d.nwb" % (lims_id, lims_id))
    save_path = os.path.join(save_dir, "%d/Data/%d_analysis.nwb" % (lims_id, lims_id))
    
    main(stimulus, nwb_path, save_path, depth)
#     try:
#         (stimulus, nwb_path, save_path, lims_id) = sys.argv[-4:]
#         main(stimulus, nwb_path, save_path, lims_id)  
#         # A /local1/cam_datasets/501836392/501836392.nwb /local1/cam_datasets/501836392/Data 501836392 True        
#         # B /local1/cam_datasets/501886692/501886692.nwb /local1/cam_datasets/501886692/Data 501886692 True
#         # C /local1/cam_datasets/501717543/501717543.nwb /local1/cam_datasets/501717543/Data 501717543 True
#     except:
#         raise(Exception('please specify stimulus A, B or C, cam_directory, lims_id'))

#    main('A', '/Users/saskiad/Documents/Data/ophysdev/502115959/502115959.nwb', 'r/Users/saskiad/Documents/Data/ophysdev/502115959/Data', '502115959')        
#    main('B', '/local1/cam_datasets/501886692/501886692.nwb', '/local1/cam_datasets/501886692/Data', '501886692')
#    main('C', '/local1/cam_datasets/501717543/501717543.nwb', '/local1/cam_datasets/501717543/Data', '501717543')
    
#
##
####            
