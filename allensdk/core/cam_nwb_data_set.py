# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:04:24 2015

@author: saskiad@alleninstitute.org
Functions to extract relevant data from the CAM NWB files
"""
import h5py
import pandas as pd
import numpy as np
import allensdk.cam.roi_masks as roi
import itertools
from collections import defaultdict
from allensdk.cam.locally_sparse_noise import LocallySparseNoise

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
        return timestamps, celltraces

    def get_corrected_fluorescence_traces(self):
        '''returns an array of fluorescence traces for all ROI and the timestamps for each datapoint'''
        f = h5py.File(self.nwb_file, 'r')
        timestamps = f['processing']['cortical_activity_map_pipeline']['Fluorescence']['ROI Masks']['timestamps'].value
        celltraces = f['processing']['cortical_activity_map_pipeline']['Fluorescence']['ROI Masks']['data'].value 
        np_traces = f['processing']['cortical_activity_map_pipeline']['Fluorescence']['ROI Masks']['neuropil_traces'].value 
        r = f['processing']['cortical_activity_map_pipeline']['Fluorescence']['ROI Masks']['r'].value 
        f.close()

        fc = celltraces - np_traces * r[:, np.newaxis]

        return timestamps, fc
        
    def get_dff_traces(self):
        '''returns an array of fluorescence traces for all ROI and the timestamps for each datapoint'''
        f = h5py.File(self.nwb_file, 'r')
        timestamps = f['processing']['cortical_activity_map_pipeline']['DfOverF']['ROI Masks']['timestamps'].value
        celltraces = f['processing']['cortical_activity_map_pipeline']['DfOverF']['ROI Masks']['data'].value 
        f.close()
        return timestamps, celltraces

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

    def get_drifting_gratings_stimulus_table(self):
        ''' TODO '''
        return self.get_abstract_feature_series_stimulus_table("drifting_gratings_stimulus")

    def get_natural_movie_stimulus_table(self, movie_name):
        ''' TODO '''
        return self.get_indexed_time_series_stimulus_table(movie_name + "_stimulus")

    def get_natural_scenes_stimulus_table(self):
        ''' TODO '''
        return self.get_indexed_time_series_stimulus_table("natural_scenes_stimulus")

    def get_static_gratings_stimulus_table(self):
        ''' TODO '''
        return self.get_abstract_feature_series_stimulus_table("static_gratings_stimulus")

    def get_locally_sparse_noise_stimulus_table(self):
        ''' TODO '''
        return self.get_indexed_time_series_stimulus_table("locally_sparse_noise_stimulus")

    def get_spontaneous_activity_stimulus_table(self):
        ''' TODO '''
        k = "stimulus/presentation/spontaneous_stimulus"
        f = h5py.File(self.nwb_file, 'r')    
        events = f[k + '/data'].value
        frame_dur = f[k + '/frame_duration'].value
        f.close()

        start_inds = np.where(events == 1)
        stop_inds = np.where(events == -1)

        if len(start_inds) != len(stop_inds):
            raise Exception("inconsistent start and time times in spontaneous activity stimulus table")

        stim_data = np.column_stack([frame_dur[start_inds,0], frame_dur[stop_inds,0]]).astype(int)

        stimulus_table = pd.DataFrame(stim_data, columns=['start','end'])

        return stimulus_table

    def get_indexed_time_series_stimulus_table(self, stimulus_name):
        ''' TODO '''
        
        k = "stimulus/presentation/%s" % stimulus_name

        f = h5py.File(self.nwb_file, 'r')    
        inds = f[k + '/data'].value
        frame_dur = f[k + '/frame_duration'].value
        f.close()

        stimulus_table = pd.DataFrame(inds, columns=['frame'])
        stimulus_table.loc[:,'start'] = frame_dur[:,0].astype(int)
        stimulus_table.loc[:,'end'] = frame_dur[:,1].astype(int)

        return stimulus_table

    def get_abstract_feature_series_stimulus_table(self, stimulus_name):
        '''returns a DataFrame of the stimulus table for a specified stimulus'''

        k = "stimulus/presentation/%s" % stimulus_name

        f = h5py.File(self.nwb_file, 'r')    
        stim_data = f[k + '/data'].value
        features = f[k + '/features'].value
        frame_dur = f[k + '/frame_duration'].value
        f.close()

        stimulus_table = pd.DataFrame(stim_data, columns=features)
        stimulus_table.loc[:,'start'] = frame_dur[:,0].astype(int)
        stimulus_table.loc[:,'end'] = frame_dur[:,1].astype(int)

        return stimulus_table

    def get_stimulus_template(self, stimulus_name):
        '''returns an array of the stimulus template for a specified stimulus'''
        stim_name = stimulus_name + "_image_stack"
        f = h5py.File(self.nwb_file, 'r')
        image_stack = f['stimulus']['templates'][stim_name]['data'].value
        f.close()
        return image_stack

    def get_locally_sparse_noise_stimulus_template(self, mask_off_screen=True):
        template = self.get_stimulus_template("locally_sparse_noise")
        
        # build mapping from template coordinates to display coordinates
        template_shape = (28, 16)
        template_display_shape = (1260, 720)
        display_shape = (1920, 1200)

        scale = [
            float(template_shape[0]) / float(template_display_shape[0]),
            float(template_shape[1]) / float(template_display_shape[1])
            ]
        offset = [
            -(display_shape[0] - template_display_shape[0]) * 0.5,
             -(display_shape[1] - template_display_shape[1]) * 0.5
             ]

        x,y = np.meshgrid(np.arange(display_shape[0]), np.arange(display_shape[1]), indexing='ij')
        template_display_coords = np.array([(x + offset[0]) * scale[0] - 0.5, 
                                            (y + offset[1]) * scale[1] - 0.5], 
                                           dtype=float)
        template_display_coords = np.rint(template_display_coords).astype(int)

        # build mask
        template_mask, template_frac = mask_stimulus_template(template_display_coords, template_shape)

        if mask_off_screen:
            template[:,~template_mask.T] = LocallySparseNoise.LSN_OFF_SCREEN

        return template, template_mask.T
    
    def get_roi_mask(self):
        '''returns an array of all the ROI masks'''
        f = h5py.File(self.nwb_file, 'r')
        mask_loc = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['ROI Masks']
        roi_list = f['processing']['cortical_activity_map_pipeline']['ImageSegmentation']['ROI Masks']['roi_list'].value
        
        roi_array = []
        for i,v in enumerate(roi_list):
            m = roi.create_roi_mask(512, 512, [0,0,0,0], pix_list=mask_loc[v]["pix_mask"].value, label=v)
            roi_array.append(m)
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


def warp_stimulus_coords(vertices,
                         distance=15.0,
                         mon_height_cm=32.5,
                         mon_width_cm=51.0,
                         mon_res=(1920,1200),
                         eyepoint=(0.5,0.5)):
    """
    For a list of screen vertices, provides a corresponding list of texture
        coordinates.

    Args:
        vertices (numpy.ndarray): [[x0,y0], [x1,y1], ...]  A set of vertices to
            convert to texture positions.
        distance (float): distance from the monitor in cm.
        mon_height_cm (float): monitor height in cm
        mon_width_cm (float): monitor width in cm
        mon_res (tuple): monitor resolution (x,y)
        eyepoint (tuple): eye position relative to monitor bottom left. center
            is (0.5, 0.5)
    """

    mon_width_cm = float(mon_width_cm)
    mon_height_cm = float(mon_height_cm)
    distance = float(distance)
    mon_res_x, mon_res_y = float(mon_res[0]), float(mon_res[1])

    vertices = vertices.astype(np.float)

    # from pixels (-1920/2 -> 1920/2) to stimulus space (-0.5->0.5)
    vertices[:, 0] = vertices[:, 0] / mon_res_x
    vertices[:, 1] = vertices[:, 1] / mon_res_y

    x = (vertices[:,0] + 0.5) * mon_width_cm
    y = (vertices[:,1] + 0.5) * mon_height_cm

    xEye = eyepoint[0] * mon_width_cm
    yEye = eyepoint[1] * mon_height_cm

    x = x - xEye
    y = y - yEye

    r = np.sqrt(np.square(x) + np.square(y) + np.square(distance))

    azimuth = np.arctan(x / distance)
    altitude = np.arcsin(y / r)

    # calculate the texture coordinates
    tx = distance * (1 + x / r) - distance
    ty = distance * (1 + y / r) - distance

    # prevent div0
    azimuth[azimuth == 0] = np.finfo(np.float32).eps
    altitude[altitude == 0] = np.finfo(np.float32).eps

    # the texture coordinates (which are now lying on the sphere)
    # need to be remapped back onto the plane of the display.
    # This effectively stretches the coordinates away from the eyepoint.

    centralAngle = np.arccos(np.cos(altitude) * np.cos(np.abs(azimuth)))
    # distance from eyepoint to texture vertex
    arcLength = centralAngle * distance
    # remap the texture coordinate
    theta = np.arctan2(ty, tx)
    tx = arcLength * np.cos(theta)
    ty = arcLength * np.sin(theta)

    u_coords = tx / mon_width_cm 
    v_coords = ty / mon_height_cm

    retCoords = np.column_stack((u_coords, v_coords))

    # back to pixels
    retCoords[:, 0] = retCoords[:, 0] * mon_res_x
    retCoords[:, 1] = retCoords[:, 1] * mon_res_y

    return retCoords

def make_display_mask(display_shape=(1920,1200)):
    x = np.array(range(display_shape[0])) - display_shape[0]/2
    y = np.array(range(display_shape[1])) - display_shape[1]/2
    display_coords = np.array(list(itertools.product(x,y)))
                                                     
    warped_coords = warp_stimulus_coords(display_coords).astype(int)

    off_warped_coords = np.array([ warped_coords[:,0] + display_shape[0]/2,
                                   warped_coords[:,1] + display_shape[1]/2 ])

    
    used_coords = set()
    for i in range(off_warped_coords.shape[1]):
        used_coords.add((off_warped_coords[0,i], off_warped_coords[1,i]))

    used_coords = ( np.array([x for (x,y) in used_coords ]),
                    np.array([y for (x,y) in used_coords ]) )

    mask = np.zeros(display_shape)
    mask[used_coords] = 1

    return mask

def mask_stimulus_template(template_display_coords, template_shape, display_mask=None, threshold=1.0):
   
    if display_mask is None:
        display_mask = make_display_mask()

    frac = np.zeros(template_shape)
    mask = np.zeros(template_shape, dtype=bool)
    for y in range(template_shape[1]):
        for x in range(template_shape[0]):
            tdcm = np.where((template_display_coords[0,:,:] == x) & (template_display_coords[1,:,:] == y))
            v = display_mask[tdcm]
            f = np.sum(v) / len(v)
            frac[x,y] = f
            mask[x,y] = f >= threshold
    
    return mask, frac

    

    
    
                      

    
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
