# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:26:59 2015

@author: saskiad
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from drifting_grating import DriftingGrating
from movie_analysis import MovieAnalysis
plt.ioff()
#import Analysis.OPTools_Nikon as op
import CAM_plotting as cp
import os


if __name__=='__main__':
    #LIMSID = '501886692'
    LIMSID = '501836392'
    Cre = 'Cux2'
    HVA = 'AL'
    depth= 175
    cam_directory = '/local1/cam_datasets'               
    #exptpath = os.path.join(r's/saskiad/Documents/Data/ophysdev', LIMSID)
    exptpath = os.path.join(cam_directory, LIMSID)
    h5path = None
    datarate = 30    
    movie_name = 'natural_movie_one'
    dg = DriftingGrating(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)

#    sg = StaticGrating(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)    
    nm1 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)
#    ni = NaturalImages(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)
#    movie_name = 'natural_movie_two'    
#    nm2 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)
    movie_name = 'natural_movie_three'
    nm3 = MovieAnalysis(exptpath, h5path, datarate, LIMSID, Cre, HVA, depth, movie_name)    
    #cp.plot_Movie_All(nm)
    #lsn = LocallySN(exptpath, h5path,datarate, LIMSID, Cre, HVA, depth)
    cp.plot_3SA(dg, nm1, nm3)
    cp.plot_Drifting_grating_Traces(dg)
    
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
#filename = 'Data.h5'
#fullfilename = os.path.join(sg.savepath, filename)
#store = pd.HDFStore(fullfilename)
#store['stim_table_sg'] = sg.stim_table
#store['sweep_response_sg'] = sg.sweep_response
#store['mean_sweep_response_sg'] = sg.mean_sweep_response
#store['sweep_response_nm1'] = nm1.sweep_response
#store['stim_table_nm1'] = nm1.stim_table
#store['sweep_response_ni'] = ni.sweep_response
#store['stim_table_ni'] = ni.stim_table
#store['mean_sweep_response_ni'] = ni.mean_sweep_response
##store['sweeptable'] = dg.sweeptable
#store.close()
#f = h5py.File(fullfilename, 'r+')
##dset6 = f.create_dataset('receptive_field', data=lsn.receptive_field)
##dset = f.create_dataset('celltraces', data=sg.celltraces)
##dset2 = f.create_dataset('twop_frames', data=sg.twop_frames)
##dset3 = f.create_dataset('acquisition_rate', data=sg.acquisition_rate)
#dset4= f.create_dataset('celltraces_dff', data=nm1.celltraces_dff)
##dset5 = f.create_dataset('dxcm', data=sg.dxcm)  
#dset6 = f.create_dataset('response_sg', data=sg.response)
#dset5 = f.create_dataset('response_ni', data=ni.response)
#dset = f.create_dataset('binned_cells_sp', data=nm1.binned_cells_sp)
#dset1 = f.create_dataset('binned_cells_vis', data=nm1.binned_cells_vis)
#dset2 = f.create_dataset('binned_dx_sp', data=nm1.binned_dx_sp)
#dset3 = f.create_dataset('binned_dx_vis',data=nm1.binned_dx_vis)
###f.keys()
#f.close()
####            
