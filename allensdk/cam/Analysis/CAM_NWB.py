# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:04:24 2015

@author: saskiad@alleninstitute.org
Functions to extract relevant data from the CAM NWB files
"""
import h5py
import pandas as pd
import numpy as np

def get_Fluorescence_Traces(NWB_file):
    '''returns an array of fluorescence traces for all ROI and the timestamps for each datapoint'''
    f = h5py.File(NWB_file, 'r')
    timestamps = f['processing']['cortical_activity_map_pipeline']['Fluorescence']['ROI Masks']['timestamps'].value
    celltraces = f['processing']['cortical_activity_map_pipeline']['Fluorescence']['ROI Masks']['data'].value 
    f.close()
    return timestamps, celltraces
    
def get_Max_Projection(NWB_file):
    '''returns the maximum projection image for the 2P data'''
    f = h5py.File(NWB_file, 'r')
    max_projection = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['ROI Masks']['reference_images']['maximum_intensity_projection_image']['data'].value
    f.close()
    return max_projection

def get_Stimulus_Table(NWB_file, stimulus_name):
    '''returns a DataFrame of the stimulus table for a specified stimulus'''
    stim_name = stimulus_name + "_stimulus"    
    f = h5py.File(NWB_file, 'r')    
    stim_data = f['stimulus']['presentation'][stim_name]['data'].value
    features=f['stimulus']['presentation'][stim_name]['features'].value
    f.close()
    
    #capitalize names of stimulus table
    Features = []
    for a in features:
        Features.append(a.capitalize())

    stimulus_table = pd.DataFrame(stim_data, columns=Features)
    stimulus_table.Start = stimulus_table.Start.astype(int)
    stimulus_table.End = stimulus_table.End.astype(int)      
    return stimulus_table

def get_Stimulus_Template(NWB_file, stimulus_name):
    '''returns an array of the stimulus template for a specified stimulus'''
    stim_name = stimulus_name + "_image_stack"
    f = h5py.File(NWB_file, 'r')
    image_stack = f['stimulus']['templates'][stim_name]['data'].value
    f.close()
    return image_stack

def get_ROI_Mask(NWB_file):
    '''returns an array of all the ROI masks'''
    f = h5py.File(NWB_file, 'r')
    mask_loc = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['ROI Masks']
    roi_list = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['ROI Masks']['roi_list'].value
    
    roi_array = np.empty((len(roi_list),512,512))    
    for i,v in enumerate(roi_list):
        roi_array[i,:,:] = mask_loc[v]['img_mask']
    f.close()
    return roi_array

def get_Meta_Data(NWB_file):
    '''returns a dictionary of meta data associated with each experiment, including Cre line, specimen number, visual area imaged, imaging depth'''
    #TODO: adapt this for current meta data    
    f = h5py.File(NWB_file, 'r')
    Cre = f['general']['specimen'].value.split('-')[0]
    specimen = f['general']['mouse_number'].value
    HVA = f['general']['hva'].value
    depth = f['general']['depth_of_imaging'].value
    system = f['general']['microscope'].value
    lims_id = f['general']['lims_id'].value
    f.close()
    meta ={'Cre': Cre, 'specimen':specimen, 'HVA':HVA, 'depth':depth, 'system':system, 'lims_id':lims_id}
    return meta

def get_Running_Speed(NWB_file):
    '''returns the mouse running speed in cm/s'''
    f = h5py.File(NWB_file, 'r')
    dxcm = f['processing']['cortical_activity_map_pipeline']['BehavioralTimeSeries']['running_speed']['data'].value
    dxtime = f['processing']['cortical_activity_map_pipeline']['BehavioralTimeSeries']['running_speed']['timestamps'].value
    timestamps = f['processing']['cortical_activity_map_pipeline']['Fluorescence']['ROI Masks']['timestamps'].value    
    f.close()   
    dxcm = dxcm[:,0]
    if dxtime[0] != timestamps[0]:
        adjust = np.where(timestamps==dxtime[0])[0][0]
        dxtime = np.insert(dxtime, 0, timestamps[:adjust])
        dxcm = np.insert(dxcm, 0, np.repeat(np.NaN, adjust))
    adjust = len(timestamps) - len(dxtime)
    if adjust>0:
        dxtime = np.append(dxtime, timestamps[(-1*adjust):])
        dxcm = np.append(dxcm, np.repeat(np.NaN, adjust))
    return dxcm, dxtime

def get_Motion_Correction(NWB_file):
    '''returns a DataFrame containing the x- and y- translation of each image used for image alignment'''
    f = h5py.File(NWB_file, 'r')
    motion_log = f['processing']['cortical_activity_map_pipeline']['MotionCorrection']['2p_image_series']['xy_translations']['data'].value
    motion_time = f['processing']['cortical_activity_map_pipeline']['MotionCorrection']['2p_image_series']['xy_translations']['timestamps'].value
    motion_names = f['processing']['cortical_activity_map_pipeline']['MotionCorrection']['2p_image_series']['xy_translations']['feature_description'].value
    motion_correction = pd.DataFrame(motion_log, columns=motion_names)
    motion_correction['timestamp'] = motion_time    
    f.close()
    return motion_correction
    
#def getMovieShape(NWB_file):
#    '''returns the shape of the hdf5 movie file'''
#    f = h5py.File(NWB_file, 'r')
#    print f['acquisition']['timeseries']['2p_image_series']['data'].shape
#    f.close()
#    return
#
#def getMovieSlice(NWB_file, t_values, x_values, y_values):
#    '''returns a slice of the hdf5 movie file. Must provide list of '''
#    f = h5py.File(NWB_file, 'r')
#    temp = f['acquisition']['timeseries']['2p_image_series']['data'][t_values,:,:]
#    temp2 = temp[:,x_values,:]
#    movie_slice = temp2[:,:,y_values]
#    f.close()
#    return movie_slice