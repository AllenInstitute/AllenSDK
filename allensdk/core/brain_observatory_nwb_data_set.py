# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:04:24 2015

@author: saskiad@alleninstitute.org
Functions to extract relevant data from the CAM NWB files
"""
import h5py
import pandas as pd
import numpy as np
import allensdk.brain_observatory.roi_masks as roi
import itertools
from collections import defaultdict
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise
import dateutil
import re

class MissingStimulusException(Exception): pass

class BrainObservatoryNwbDataSet(object):
    MOVIE_FOV_PX = (512, 512)
    PIPELINE_DATASET = 'brain_observatory_pipeline'
    
    FILE_METADATA_MAPPING = { 
        'age': 'general/subject/age',
        'sex': 'general/subject/sex',
        'imaging_depth': 'general/optophysiology/imaging_plane_1/imaging depth',
        'targeted_structure': 'general/optophysiology/imaging_plane_1/location',
        'ophys_experiment_id': 'general/session_id',
        'experiment_container_id': 'general/experiment_container_id',
        'device_string': 'general/devices/2-photon microscope',
        'excitation_lambda': 'general/optophysiology/imaging_plane_1/excitation_lambda',
        'indicator': 'general/optophysiology/imaging_plane_1/indicator',
        'fov': 'general/fov',
        'genotype': 'general/subject/genotype',
        'session_start_time': 'session_start_time',
        'session_type': 'general/session_type'
        }

    STIMULUS_TABLE_TYPES = {
        'abstract_feature_series': [ 'drifting_gratings', 'static_gratings' ],
        'indexed_time_series': [ 'natural_movie_one', 'natural_movie_two', 'natural_movie_three', 
                                 'natural_scenes', 'locally_sparse_noise' ]
        }
    
    def __init__(self, nwb_file):
        self.nwb_file = nwb_file
        

    def get_fluorescence_traces(self, cell_specimen_ids=None):
        ''' Returns an array of fluorescence traces for all ROI and 
        the timestamps for each datapoint

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then all are returned

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        traces: 2D numpy array
            Fluorescence traces for each cell
        '''
        all_cell_specimen_ids = list(self.get_cell_specimen_ids())

        with h5py.File(self.nwb_file, 'r') as f:
            timestamps = f['processing'][self.PIPELINE_DATASET]['Fluorescence']['imaging_plane_1']['timestamps'].value
            ds = f['processing'][self.PIPELINE_DATASET]['Fluorescence']['imaging_plane_1']['data']

            if cell_specimen_ids is None:
                cell_traces = ds.value 
            else:
                inds = [ all_cell_specimen_ids.index(i) for i in cell_specimen_ids ]
                cell_traces = ds[inds,:]

        return timestamps, cell_traces

    def get_neuropil_traces(self, cell_specimen_ids=None):
        ''' Returns an array of fluorescence traces for all ROIs 
        and the timestamps for each datapoint

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then all are returned

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        traces: 2D numpy array
            Neuropil fluorescence traces for each cell
        '''
        all_cell_specimen_ids = self.get_cell_specimen_ids()

        with h5py.File(self.nwb_file, 'r') as f:
            timestamps = f['processing'][self.PIPELINE_DATASET]['Fluorescence']['imaging_plane_1']['timestamps'].value

            ds = f['processing'][self.PIPELINE_DATASET]['Fluorescence']['imaging_plane_1']['neuropil_traces']
            if cell_specimen_ids is None:
                np_traces = ds.value 
            else:
                inds = [ list(all_cell_specimen_ids).index(i) for i in cell_specimen_ids ]
                np_traces = ds[inds,:]

        return timestamps, np_traces

    def get_corrected_fluorescence_traces(self, cell_specimen_ids=None):
        ''' Returns an array of neuropil-corrected fluorescence traces 
        for all ROIs and the timestamps for each datapoint

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then all are returned

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        traces: 2D numpy array
            Corrected fluorescence traces for each cell
        '''
        all_cell_specimen_ids = list(self.get_cell_specimen_ids())

        with h5py.File(self.nwb_file, 'r') as f:
            timestamps = f['processing'][self.PIPELINE_DATASET]['Fluorescence']['imaging_plane_1']['timestamps'].value
            cell_traces_ds = f['processing'][self.PIPELINE_DATASET]['Fluorescence']['imaging_plane_1']['data']
            np_traces_ds = f['processing'][self.PIPELINE_DATASET]['Fluorescence']['imaging_plane_1']['neuropil_traces']
            r_ds = f['processing'][self.PIPELINE_DATASET]['Fluorescence']['imaging_plane_1']['r']

            if cell_specimen_ids is None:
                cell_traces = cell_traces_ds.value 
                np_traces = np_traces_ds.value 
                r = r_ds.value 
            else:
                inds = [ all_cell_specimen_ids.index(i) for i in cell_specimen_ids ]
                cell_traces = cell_traces_ds[inds,:]
                np_traces = np_traces_ds[inds,:]
                r = r_ds[inds]

        fc = cell_traces - np_traces * r[:, np.newaxis]

        return timestamps, fc

        
    def get_dff_traces(self, cell_specimen_ids=None):
        ''' Returns an array of dF/F traces for all ROIs and 
        the timestamps for each datapoint

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return data for. If this is None (default)
            then all are returned

        Returns
        -------
        timestamps: 2D numpy array
            Timestamp for each fluorescence sample

        dF/F: 2D numpy array
            dF/F values for each cell
        '''
        all_cell_specimen_ids = list(self.get_cell_specimen_ids())

        with h5py.File(self.nwb_file, 'r') as f:
            timestamps = f['processing'][self.PIPELINE_DATASET]['DfOverF']['imaging_plane_1']['timestamps'].value

            if cell_specimen_ids is None:
                cell_traces = f['processing'][self.PIPELINE_DATASET]['DfOverF']['imaging_plane_1']['data'].value 
            else:
                inds = [ all_cell_specimen_ids.index(i) for i in cell_specimen_ids ]
                cell_traces = f['processing'][self.PIPELINE_DATASET]['DfOverF']['imaging_plane_1']['data'][inds,:]

        return timestamps, cell_traces


    def get_roi_ids(self):
        ''' Returns an array of IDs for all ROIs in the file

        Returns
        -------
        ROI IDs: list
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            roi_id = f['processing'][self.PIPELINE_DATASET]['ImageSegmentation']['roi_ids'].value 
        return roi_id

    def get_cell_specimen_ids(self):
        ''' Returns an array of cell IDs for all cells in the file

        Returns
        -------
        cell specimen IDs: list
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            cell_id = f['processing'][self.PIPELINE_DATASET]['ImageSegmentation']['cell_specimen_ids'].value 
        return cell_id

    def get_session_type(self):
        ''' Returns the type of experimental session, presently one of the
        following: three_session_A, three_session_B, three_session_C

        Returns
        -------
        session type: string
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            session_type = f['general/session_type'].value
        return session_type

        
    def get_max_projection(self):
        '''Returns the maximum projection image for the 2P movie.
        
        Returns
        -------
        max projection: np.ndarray
        '''

        with h5py.File(self.nwb_file, 'r') as f:
            max_projection = f['processing'][self.PIPELINE_DATASET]['ImageSegmentation']['imaging_plane_1']['reference_images']['maximum_intensity_projection_image']['data'].value
        return max_projection

    def list_stimuli(self):
        ''' Return a list of the stimuli presented in the experiment. 
        
        Returns
        -------
        stimuli: list of strings
        '''

        with h5py.File(self.nwb_file, 'r') as f:
            keys = f["stimulus/presentation/"].keys()
        return [ k.replace('_stimulus','') for k in keys ]

    def get_stimulus_table(self, stimulus_name):
        ''' Return a stimulus table given a stimulus name '''
        if stimulus_name in self.STIMULUS_TABLE_TYPES['abstract_feature_series']:
            return self.get_abstract_feature_series_stimulus_table(stimulus_name + "_stimulus")
        elif stimulus_name in self.STIMULUS_TABLE_TYPES['indexed_time_series']:
            return self.get_indexed_time_series_stimulus_table(stimulus_name + "_stimulus")
        elif stimulus_name == 'spontaneous':
            return self.get_spontaneous_activity_stimulus_table()
        else:
            raise IOError("Could not find a stimulus table named '%s'" % stimulus_name)

    def get_spontaneous_activity_stimulus_table(self):
        ''' Return the spontaneous activity stimulus table, if it exists.

        Returns
        -------
        stimulus table: pd.DataFrame
        '''
        k = "stimulus/presentation/spontaneous_stimulus"
        with h5py.File(self.nwb_file, 'r') as f:
            events = f[k + '/data'].value
            frame_dur = f[k + '/frame_duration'].value

        start_inds = np.where(events == 1)
        stop_inds = np.where(events == -1)

        if len(start_inds) != len(stop_inds):
            raise Exception("inconsistent start and time times in spontaneous activity stimulus table")

        stim_data = np.column_stack([frame_dur[start_inds,0].T, frame_dur[stop_inds,0].T]).astype(int)

        stimulus_table = pd.DataFrame(stim_data, columns=['start','end'])

        return stimulus_table

    def get_indexed_time_series_stimulus_table(self, stimulus_name):
        ''' Return the a stimulus table for an indexed time series.

        Returns
        -------
        stimulus table: pd.DataFrame
        '''
        
        k = "stimulus/presentation/%s" % stimulus_name

        with h5py.File(self.nwb_file, 'r') as f: 
            if k not in f:
                raise MissingStimulusException("Stimulus not found: %s" % stimulus_name)    
            inds = f[k + '/data'].value
            frame_dur = f[k + '/frame_duration'].value

        stimulus_table = pd.DataFrame(inds, columns=['frame'])
        stimulus_table.loc[:,'start'] = frame_dur[:,0].astype(int)
        stimulus_table.loc[:,'end'] = frame_dur[:,1].astype(int)

        return stimulus_table

    def get_abstract_feature_series_stimulus_table(self, stimulus_name):
        ''' Return the a stimulus table for an abstract feature series.

        Returns
        -------
        stimulus table: pd.DataFrame
        '''

        k = "stimulus/presentation/%s" % stimulus_name

        with h5py.File(self.nwb_file, 'r') as f:
            if k not in f:
                raise MissingStimulusException("Stimulus not found: %s" % stimulus_name)
            stim_data = f[k + '/data'].value
            features = f[k + '/features'].value
            frame_dur = f[k + '/frame_duration'].value

        stimulus_table = pd.DataFrame(stim_data, columns=features)
        stimulus_table.loc[:,'start'] = frame_dur[:,0].astype(int)
        stimulus_table.loc[:,'end'] = frame_dur[:,1].astype(int)

        return stimulus_table

    def get_stimulus_template(self, stimulus_name):
        ''' Return an array of the stimulus template for the specified stimulus.

        Parameters
        ----------
        stimulus_name: string
            Must be one of the strings returned by list_stimuli().

        Returns
        -------
        stimulus table: pd.DataFrame
        '''
        stim_name = stimulus_name + "_image_stack"
        with h5py.File(self.nwb_file, 'r') as f:
            image_stack = f['stimulus']['templates'][stim_name]['data'].value
        return image_stack

    def get_locally_sparse_noise_stimulus_template(self, mask_off_screen=True):
        ''' Return an array of the stimulus template for the specified stimulus.

        Parameters
        ----------
        mask_off_screen: boolean
           Set off-screen regions of the stimulus to LocallySparseNoise.LSN_OFF_SCREEN.

        Returns
        -------
        tuple: (template, off-screen mask)
        '''
        
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
    
    def get_roi_mask(self, cell_specimen_ids=None):
        ''' Returns an array of all the ROI masks

        Parameters
        ----------
        cell specimen IDs: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then all are returned

        Returns
        -------
            List of ROI_Mask objects
        '''

        all_cell_specimen_ids = self.get_cell_specimen_ids()

        with h5py.File(self.nwb_file, 'r') as f:
            mask_loc = f['processing'][self.PIPELINE_DATASET]['ImageSegmentation']['imaging_plane_1']
            roi_list = f['processing'][self.PIPELINE_DATASET]['ImageSegmentation']['imaging_plane_1']['roi_list'].value
        
            inds = None
            if cell_specimen_ids is None:
                inds = range(len(all_cell_specimen_ids))
            else:
                inds = [ list(all_cell_specimen_ids).index(i) for i in cell_specimen_ids ]

            roi_array = []
            for i in inds:
                v = roi_list[i]
                m = roi.create_roi_mask(self.MOVIE_FOV_PX[0], self.MOVIE_FOV_PX[1], [0,0,0,0], pix_list=mask_loc[v]["pix_mask"].value, label=v)
                roi_array.append(m)

        return roi_array
    
    def get_metadata(self):
        ''' Returns a dictionary of meta data associated with each 
        experiment, including Cre line, specimen number, 
        visual area imaged, imaging depth

        Returns
        -------
        metadata: dictionary
        '''
        
        meta = {}
            
        with h5py.File(self.nwb_file, 'r') as f:
            for memory_key, disk_key in BrainObservatoryNwbDataSet.FILE_METADATA_MAPPING.items():
                v = f[disk_key].value
                if v.dtype.type is np.string_:
                    v = str(v)
                meta[memory_key] = v

        meta['cre_line'] = meta['genotype'].split(';')[0]
        meta['imaging_depth_um'] = int(meta['imaging_depth'].split()[0])
        del meta['imaging_depth']
        meta['ophys_experiment_id'] = int(meta['ophys_experiment_id'])
        meta['experiment_container_id'] = int(meta['experiment_container_id'])
        meta['session_start_time'] = dateutil.parser.parse(meta['session_start_time'])

        # parse the age in days
        m = re.match("(.*?) days", meta['age'])
        if m:
            meta['age_days'] = int(m.groups()[0])
            del meta['age']
        else:
            raise IOError("Could not find device.")

        # parse the device string (ugly, sorry)
        device_string = meta['device_string']
        del meta['device_string']

        m = re.match("(.*?)\.\s(.*?)\sPlease*", device_string)
        if m:
            device, device_name = m.groups()
            meta['device'] = device
            meta['device_name'] = device_name
        else:
            raise IOError("Could not find device.")

        return meta
        

    
    def get_running_speed(self):
        ''' Returns the mouse running speed in cm/s
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            dxcm = f['processing'][self.PIPELINE_DATASET]['BehavioralTimeSeries']['running_speed']['data'].value
            dxtime = f['processing'][self.PIPELINE_DATASET]['BehavioralTimeSeries']['running_speed']['timestamps'].value
            timestamps = f['processing'][self.PIPELINE_DATASET]['Fluorescence']['imaging_plane_1']['timestamps'].value    

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
        ''' Returns a Panda DataFrame containing the x- and y- translation of each image used for image alignment
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            motion_log = f['processing'][self.PIPELINE_DATASET]['MotionCorrection']['2p_image_series']['xy_translations']['data'].value
            motion_time = f['processing'][self.PIPELINE_DATASET]['MotionCorrection']['2p_image_series']['xy_translations']['timestamps'].value
            motion_names = f['processing'][self.PIPELINE_DATASET]['MotionCorrection']['2p_image_series']['xy_translations']['feature_description'].value
            motion_correction = pd.DataFrame(motion_log, columns=motion_names)
            motion_correction['timestamp'] = motion_time    

        return motion_correction
    
    
    def save_analysis_dataframes(self, *tables):
        # NOTE: should use NWB library to write data to NWB file. It is 
        #   designed to avoid possible corruption of the file in event
        #   of a failed write
        store = pd.HDFStore(self.nwb_file, mode='a')

        for k,v in tables:
            store.put('analysis/%s' % (k), v)

        store.close()
        

    def save_analysis_arrays(self, *datasets):    
        # NOTE: should use NWB library to write data to NWB file. It is 
        #   designed to avoid possible corruption of the file in event
        #   of a failed write
        with h5py.File(self.nwb_file, 'a') as f:
            for k,v in datasets:
                if k in f['analysis']:
                    del f['analysis'][k]
                f.create_dataset('analysis/%s' % k, data=v)


def warp_stimulus_coords(vertices,
                         distance=15.0,
                         mon_height_cm=32.5,
                         mon_width_cm=51.0,
                         mon_res=(1920,1200),
                         eyepoint=(0.5,0.5)):
    """  
    For a list of screen vertices, provides a corresponding list of texture coordinates.

    Parameters
    ----------
    vertices: numpy.ndarray
        [[x0,y0], [x1,y1], ...] A set of vertices to  convert to texture positions.
    distance: float
        distance from the monitor in cm.
    mon_height_cm: float
        monitor height in cm
    mon_width_cm: float
        monitor width in cm
    mon_res: tuple
        monitor resolution (x,y)
    eyepoint: tuple

    Returns
    -------
    np.ndarray
        x,y coordinates shaped like the input that describe what pixel coordinates
        are displayed an the input coordinates after warping the stimulus.
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
    ''' Build a display-shaped mask that indicates which pixels are on screen after warping the stimulus. '''
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
    ''' Build a mask for a stimulus template of a given shape and display coordinates that indicates
    which part of the template is on screen after warping.

    Parameters
    ----------
    template_display_coords: list
        list of (x,y) display coordinates

    template_shape: tuple
        (width,height) of the display template  

    display_mask: np.ndarray
        boolean 2D mask indicating which display coordinates are on screen after warping.

    threshold: float
        Fraction of pixels associated with a template display coordinate that should remain
        on screen to count as belonging to the mask. 

    Returns
    -------
    tuple: (template mask, pixel fraction)
    '''
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

                      

    
