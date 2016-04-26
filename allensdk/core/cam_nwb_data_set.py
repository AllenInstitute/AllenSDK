# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:04:24 2015

@author: saskiad@alleninstitute.org
Functions to extract relevant data from the CAM NWB files
"""
import h5py
import pandas as pd
import numpy as np

class CamNwbDataSet(object):
    file_metadata_mapping = { 'specimen': 'specimen',
                              'area': 'area_targeted',
                              'depth': 'depth_of_imaging',
                              'system': 'microscope',
                              'experiment_id': 'lims_id' }
    
    def __init__(self, nwb_file):
        self.nwb_file = nwb_file
        

    def get_fluorescence_traces(self):
        '''returns an array of fluorescence traces for all ROI and the timestamps for each datapoint'''
        f = h5py.File(self.nwb_file, 'r')
        timestamps = f['processing']['cortical_activity_map_pipeline']['Fluorescence']['ROI Masks']['timestamps'].value
        celltraces = f['processing']['cortical_activity_map_pipeline']['Fluorescence']['ROI Masks']['data'].value 
        f.close()
        return timestamps, celltraces, roi_id, cell_id
        
    def get_dff_traces(self):
        '''returns an array of fluorescence traces for all ROI and the timestamps for each datapoint'''
        f = h5py.File(self.nwb_file, 'r')
        timestamps = f['processing']['cortical_activity_map_pipeline']['DfOverF']['ROI Masks']['timestamps'].value
        celltraces = f['processing']['cortical_activity_map_pipeline']['DfOverF']['ROI Masks']['data'].value 
        f.close()
        return timestamps, celltraces, roi_id, cell_id

    def get_roi_ids(self):
        f = h5py.File(self.nwb_file, 'r')
        roi_id = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['roi_ids'].value 
        f.close()
        return roi_id

    def get_cell_specimen_ids(self):
        f = h5py.File(self.nwb_file, 'r')
        cell_id = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['cell_specimen_ids'].value 
        f.close()
        return cell_id
        
    def get_max_projection(self):
        '''returns the maximum projection image for the 2P data'''
        f = h5py.File(self.nwb_file, 'r')
        max_projection = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['ROI Masks']['reference_images']['maximum_intensity_projection_image']['data'].value
        f.close()
        return max_projection
    
    def get_stimulus_table(self, stimulus_name):
        '''returns a DataFrame of the stimulus table for a specified stimulus'''
        stim_name = stimulus_name + "_stimulus"    
        f = h5py.File(self.nwb_file, 'r')    
        stim_data = f['stimulus']['presentation'][stim_name]['data'].value
        features=f['stimulus']['presentation'][stim_name]['features'].value
        f.close()
    
        stimulus_table = pd.DataFrame(stim_data, columns=features)
        stimulus_table.start = stimulus_table.start.astype(int)
        stimulus_table.end = stimulus_table.end.astype(int)      
        return stimulus_table
    
    def get_stimulus_template(self, stimulus_name):
        '''returns an array of the stimulus template for a specified stimulus'''
        stim_name = stimulus_name + "_image_stack"
        f = h5py.File(self.nwb_file, 'r')
        image_stack = f['stimulus']['templates'][stim_name]['data'].value
        f.close()
        return image_stack
    
    def get_roi_mask(self):
        '''returns an array of all the ROI masks'''
        f = h5py.File(self.nwb_file, 'r')
        mask_loc = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['ROI Masks']
        roi_list = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['ROI Masks']['roi_list'].value
        
        roi_array = np.empty((len(roi_list),512,512))    
        for i,v in enumerate(roi_list):
            roi_array[i,:,:] = mask_loc[v]['img_mask']
        f.close()
        return roi_array
    
    def get_meta_data(self):
        '''returns a dictionary of meta data associated with each experiment, including Cre line, specimen number, visual area imaged, imaging depth'''
        #TODO: adapt this for current meta data
        
        meta = {}
            
        with h5py.File(self.nwb_file, 'r') as f:
            for memory_key, disk_key in CamNwbDataSet.file_metadata_mapping.items():
                try:
                    meta[memory_key] = f['general'][disk_key].value
                except:
                    meta[memory_key] = None

        try:
            meta['Cre'] = meta['specimen'].split('-')[0]
            meta['specimen'] = meta['specimen'].split('-')[-1]            
        except:
            meta['Cre'] = None
            meta['specimen'] = None

        return meta
        

    
    def get_running_speed(self):
        '''returns the mouse running speed in cm/s'''
        f = h5py.File(self.nwb_file, 'r')
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
    
    def get_motion_correction(self):
        '''returns a DataFrame containing the x- and y- translation of each image used for image alignment'''
        f = h5py.File(self.nwb_file, 'r')
        motion_log = f['processing']['cortical_activity_map_pipeline']['MotionCorrection']['2p_image_series']['xy_translations']['data'].value
        motion_time = f['processing']['cortical_activity_map_pipeline']['MotionCorrection']['2p_image_series']['xy_translations']['timestamps'].value
        motion_names = f['processing']['cortical_activity_map_pipeline']['MotionCorrection']['2p_image_series']['xy_translations']['feature_description'].value
        motion_correction = pd.DataFrame(motion_log, columns=motion_names)
        motion_correction['timestamp'] = motion_time    
        f.close()
        return motion_correction
    
    
    def save_analysis_dataframes(self, *tables):
        store = pd.HDFStore(self.nwb_file, mode='a')

        for k,v in tables:
            store.put('analysis/%s' % (k), v)

        store.close()
        

    def save_analysis_arrays(self, *datasets):    
        f = h5py.File(self.nwb_file, 'a')
        
        for k,v in datasets:
            if k in f['analysis']:
                del f['analysis'][k]
            f.create_dataset('analysis/%s' % k, data=v)
        
        ##f.keys()
        f.close()

    
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
