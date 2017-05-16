# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import h5py
import logging
import pandas as pd
import numpy as np
import allensdk.brain_observatory.roi_masks as roi
import itertools
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise
import allensdk.brain_observatory.stimulus_info as si
import dateutil
import re
import os
from pkg_resources import parse_version
from allensdk.brain_observatory.brain_observatory_exceptions import (MissingStimulusException,
                                                                     NoEyeTrackingException)

class BrainObservatoryNwbDataSet(object):
    PIPELINE_DATASET = 'brain_observatory_pipeline'
    SUPPORTED_PIPELINE_VERSION = "2.0"

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
        'session_type': 'general/session_type',
        'specimen_name': 'general/specimen_name',
        'generated_by': 'general/generated_by'
    }

    STIMULUS_TABLE_TYPES = {
        'abstract_feature_series': [si.DRIFTING_GRATINGS, si.STATIC_GRATINGS],
        'indexed_time_series': [si.NATURAL_MOVIE_ONE, si.NATURAL_MOVIE_TWO, si.NATURAL_MOVIE_THREE,
                                si.NATURAL_SCENES, si.LOCALLY_SPARSE_NOISE, 
                                si.LOCALLY_SPARSE_NOISE_4DEG, si.LOCALLY_SPARSE_NOISE_8DEG]

    }

    # this array was moved before file versioning was in place
    MOTION_CORRECTION_DATASETS = [ "MotionCorrection/2p_image_series/xy_translations", 
                                   "MotionCorrection/2p_image_series/xy_translation" ]

    def __init__(self, nwb_file):

        self.nwb_file = nwb_file
        self.pipeline_version = None
        
        if os.path.exists(self.nwb_file):
            meta = self.get_metadata()
            if meta and 'pipeline_version' in meta:
                pipeline_version_str = meta['pipeline_version']
                self.pipeline_version = parse_version(pipeline_version_str)

                if self.pipeline_version > parse_version(self.SUPPORTED_PIPELINE_VERSION):
                    logging.warning("File %s has a pipeline version newer than the version supported by this class (%s vs %s)."
                                    " Please update your AllenSDK." % (nwb_file, pipeline_version_str, self.SUPPORTED_PIPELINE_VERSION))



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
        timestamps = self.get_fluorescence_timestamps()
        with h5py.File(self.nwb_file, 'r') as f:
            ds = f['processing'][self.PIPELINE_DATASET][
                'Fluorescence']['imaging_plane_1']['data']

            if cell_specimen_ids is None:
                cell_traces = ds.value
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                cell_traces = ds[inds, :]

        return timestamps, cell_traces

    def get_fluorescence_timestamps(self):
        ''' Returns an array of timestamps in seconds for the fluorescence traces '''

        with h5py.File(self.nwb_file, 'r') as f:
            timestamps = f['processing'][self.PIPELINE_DATASET][
                'Fluorescence']['imaging_plane_1']['timestamps'].value
        return timestamps

    def get_neuropil_traces(self, cell_specimen_ids=None):
        ''' Returns an array of neuropil fluorescence traces for all ROIs
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

        timestamps = self.get_fluorescence_timestamps()

        with h5py.File(self.nwb_file, 'r') as f:
            if self.pipeline_version >= parse_version("2.0"):
                ds = f['processing'][self.PIPELINE_DATASET][
                    'Fluorescence']['imaging_plane_1_neuropil_response']['data']
            else:
                ds = f['processing'][self.PIPELINE_DATASET][
                    'Fluorescence']['imaging_plane_1']['neuropil_traces']

            if cell_specimen_ids is None:
                np_traces = ds.value
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                np_traces = ds[inds, :]

        return timestamps, np_traces


    def get_neuropil_r(self, cell_specimen_ids=None):
        ''' Returns a scalar value of r for neuropil correction of flourescence traces

        Parameters
        ----------
        cell_specimen_ids: list or array (optional)
            List of cell IDs to return traces for. If this is None (default)
            then results for all are returned

        Returns
        -------
        r: 1D numpy array, len(r)=len(cell_specimen_ids)
            Scalar for neuropil subtraction for each cell
        '''

        with h5py.File(self.nwb_file, 'r') as f:
            if self.pipeline_version >= parse_version("2.0"):
                r_ds = f['processing'][self.PIPELINE_DATASET][
                    'Fluorescence']['imaging_plane_1_neuropil_response']['r']
            else:
                r_ds = f['processing'][self.PIPELINE_DATASET][
                    'Fluorescence']['imaging_plane_1']['r']

            if cell_specimen_ids is None:
                r = r_ds.value
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                r = r_ds[inds]

        return r

    def get_demixed_traces(self, cell_specimen_ids=None):
        ''' Returns an array of demixed fluorescence traces for all ROIs
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
            Demixed fluorescence traces for each cell
        '''

        timestamps = self.get_fluorescence_timestamps()

        with h5py.File(self.nwb_file, 'r') as f:
            ds = f['processing'][self.PIPELINE_DATASET][
                'Fluorescence']['imaging_plane_1_demixed_signal']['data']
            if cell_specimen_ids is None:
                traces = ds.value
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                traces = ds[inds, :]

        return timestamps, traces

    def get_corrected_fluorescence_traces(self, cell_specimen_ids=None):
        ''' Returns an array of demixed and neuropil-corrected fluorescence traces
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

        # starting in version 2.0, neuropil correction follows trace demixing
        if self.pipeline_version >= parse_version("2.0"):
            timestamps, cell_traces = self.get_demixed_traces(cell_specimen_ids)
        else:
            timestamps, cell_traces = self.get_fluorescence_traces(cell_specimen_ids)

        r = self.get_neuropil_r(cell_specimen_ids)

        _, neuropil_traces = self.get_neuropil_traces(cell_specimen_ids)

        fc = cell_traces - neuropil_traces * r[:, np.newaxis]

        return timestamps, fc

    def get_cell_specimen_indices(self, cell_specimen_ids=None):
        ''' Given a list of cell specimen ids, return their index based on their order in this file.

        Parameters
        ----------
        cell_specimen_ids: list of cell specimen ids

        '''

        all_cell_specimen_ids = list(self.get_cell_specimen_ids())

        try:
            inds = [list(all_cell_specimen_ids).index(i)
                    for i in cell_specimen_ids]
        except ValueError as e:
            raise ValueError("Cell specimen not found (%s)" % str(e))

        return inds

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
        with h5py.File(self.nwb_file, 'r') as f:
            dff_ds = f['processing'][self.PIPELINE_DATASET][
                'DfOverF']['imaging_plane_1']

            timestamps = dff_ds['timestamps'].value

            if cell_specimen_ids is None:
                cell_traces = dff_ds['data'].value
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)
                cell_traces = dff_ds['data'][inds, :]

        return timestamps, cell_traces

    def get_roi_ids(self):
        ''' Returns an array of IDs for all ROIs in the file

        Returns
        -------
        ROI IDs: list
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            roi_id = f['processing'][self.PIPELINE_DATASET][
                'ImageSegmentation']['roi_ids'].value
        return roi_id

    def get_cell_specimen_ids(self):
        ''' Returns an array of cell IDs for all cells in the file

        Returns
        -------
        cell specimen IDs: list
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            cell_id = f['processing'][self.PIPELINE_DATASET][
                'ImageSegmentation']['cell_specimen_ids'].value
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
            max_projection = f['processing'][self.PIPELINE_DATASET]['ImageSegmentation'][
                'imaging_plane_1']['reference_images']['maximum_intensity_projection_image']['data'].value
        return max_projection

    def list_stimuli(self):
        ''' Return a list of the stimuli presented in the experiment.

        Returns
        -------
        stimuli: list of strings
        '''

        with h5py.File(self.nwb_file, 'r') as f:
            keys = f["stimulus/presentation/"].keys()
        return [k.replace('_stimulus', '') for k in keys]

    def get_stimulus_table(self, stimulus_name):
        ''' Return a stimulus table given a stimulus name '''
        if stimulus_name in self.STIMULUS_TABLE_TYPES['abstract_feature_series']:
            return _get_abstract_feature_series_stimulus_table(self.nwb_file, stimulus_name + "_stimulus")
        elif stimulus_name in self.STIMULUS_TABLE_TYPES['indexed_time_series']:
            try:
                return _get_indexed_time_series_stimulus_table(self.nwb_file, stimulus_name + "_stimulus")
            except:
                return _get_indexed_time_series_stimulus_table(self.nwb_file, stimulus_name)
        elif stimulus_name == 'spontaneous':
            return self.get_spontaneous_activity_stimulus_table()
        else:
            raise IOError(
                "Could not find a stimulus table named '%s'" % stimulus_name)

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
            raise Exception(
                "inconsistent start and time times in spontaneous activity stimulus table")

        stim_data = np.column_stack(
            [frame_dur[start_inds, 0].T, frame_dur[stop_inds, 0].T]).astype(int)

        stimulus_table = pd.DataFrame(stim_data, columns=['start', 'end'])

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

    def get_locally_sparse_noise_stimulus_template(self, 
                                                   stimulus, 
                                                   mask_off_screen=True):
        ''' Return an array of the stimulus template for the specified stimulus.

        Parameters
        ----------
        stimulus: string
           Which locally sparse noise stimulus to retrieve.  Must be one of: 
               stimulus_info.LOCALLY_SPARSE_NOISE
               stimulus_info.LOCALLY_SPARSE_NOISE_4DEG
               stimulus_info.LOCALLY_SPARSE_NOISE_8DEG

        mask_off_screen: boolean
           Set off-screen regions of the stimulus to LocallySparseNoise.LSN_OFF_SCREEN.

        Returns
        -------
        tuple: (template, off-screen mask)
        '''

        if stimulus not in si.LOCALLY_SPARSE_NOISE_DIMENSIONS:
            raise KeyError("%s is not a known locally sparse noise stimulus" % stimulus)

        template = self.get_stimulus_template(stimulus)

        # build mapping from template coordinates to display coordinates
        template_shape = si.LOCALLY_SPARSE_NOISE_DIMENSIONS[stimulus]
        template_shape = [ template_shape[1], template_shape[0] ]

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

        x, y = np.meshgrid(np.arange(display_shape[0]), np.arange(
            display_shape[1]), indexing='ij')
        template_display_coords = np.array([(x + offset[0]) * scale[0] - 0.5,
                                            (y + offset[1]) * scale[1] - 0.5],
                                           dtype=float)
        template_display_coords = np.rint(template_display_coords).astype(int)

        # build mask
        template_mask, template_frac = mask_stimulus_template(
            template_display_coords, template_shape)

        if mask_off_screen:
            template[:, ~template_mask.T] = LocallySparseNoise.LSN_OFF_SCREEN

        return template, template_mask.T

    def get_roi_mask_array(self, cell_specimen_ids=None):
        ''' Return a numpy array containing all of the ROI masks for requested cells.
        If cell_specimen_ids is omitted, return all masks.

        Parameters
        ----------
        cell_specimen_ids: list
            List of cell specimen ids.  Default None.

        Returns
        -------
        np.ndarray: NxWxH array, where N is number of cells
        '''

        roi_masks = self.get_roi_mask(cell_specimen_ids)

        if len(roi_masks) == 0:
            raise IOError("no masks found for given cell specimen ids")

        roi_arr = roi.create_roi_mask_array(roi_masks)

        return roi_arr

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

        with h5py.File(self.nwb_file, 'r') as f:
            mask_loc = f['processing'][self.PIPELINE_DATASET][
                'ImageSegmentation']['imaging_plane_1']
            roi_list = f['processing'][self.PIPELINE_DATASET][
                'ImageSegmentation']['imaging_plane_1']['roi_list'].value

            inds = None
            if cell_specimen_ids is None:
                inds = range(self.number_of_cells)
            else:
                inds = self.get_cell_specimen_indices(cell_specimen_ids)

            roi_array = []
            for i in inds:
                v = roi_list[i]
                roi_mask = mask_loc[v]["img_mask"].value
                m = roi.create_roi_mask(roi_mask.shape[1], roi_mask.shape[0],
                                        [0, 0, 0, 0], roi_mask=roi_mask, label=v)
                roi_array.append(m)

        return roi_array

    @property
    def number_of_cells(self):
        '''Number of cells in the experiment'''

        # Replace here is there is a better way to get this info:
        return len(self.get_cell_specimen_ids())


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
                try:
                    v = f[disk_key].value

                    # convert numpy strings to python strings
                    if v.dtype.type is np.string_:
                        if len(v.shape) == 0:
                            v = v.decode('UTF-8')
                        elif len(v.shape) == 1:
                            v = [ s.decode('UTF-8') for s in v ]
                        else:
                            raise Exception("Unrecognized metadata formatting for field %s" % disk_key)

                    meta[memory_key] = v
                except KeyError as e:
                    logging.warning("could not find key %s", disk_key)

        # extract cre line from genotype string
        genotype = meta.get('genotype')
        meta['cre_line'] = meta['genotype'].split(';')[0] if genotype else None

        imaging_depth = meta.pop('imaging_depth', None)
        meta['imaging_depth_um'] = int(imaging_depth.split()[0]) if imaging_depth else None
        
        ophys_experiment_id = meta.get('ophys_experiment_id')
        meta['ophys_experiment_id'] = int(ophys_experiment_id) if ophys_experiment_id else None

        experiment_container_id = meta.get('experiment_container_id')
        meta['experiment_container_id'] = int(experiment_container_id) if experiment_container_id else None

        # convert start time to a date object
        session_start_time = meta.get('session_start_time')
        if isinstance(session_start_time, basestring):
            meta['session_start_time'] = dateutil.parser.parse(session_start_time)

        age = meta.pop('age', None)
        if age:
            # parse the age in days
            m = re.match("(.*?) days", age)
            if m:
                meta['age_days'] = int(m.groups()[0])
            else:
                raise IOError("Could not parse age.")
            

        # parse the device string (ugly, sorry)
        device_string = meta.pop('device_string', None)
        if device_string:
            m = re.match("(.*?)\.\s(.*?)\sPlease*", device_string)
            if m:
                device, device_name = m.groups()
                meta['device'] = device
                meta['device_name'] = device_name
            else:
                raise IOError("Could not parse device string.")

        # file version
        generated_by = meta.pop('generated_by', None)
        version = generated_by[-1] if generated_by else "0.9"
        meta["pipeline_version"] = version

        return meta

    def get_running_speed(self):
        ''' Returns the mouse running speed in cm/s
        '''
        with h5py.File(self.nwb_file, 'r') as f:
            dx_ds = f['processing'][self.PIPELINE_DATASET][
                'BehavioralTimeSeries']['running_speed']
            dxcm = dx_ds['data'].value
            dxtime = dx_ds['timestamps'].value

        timestamps = self.get_fluorescence_timestamps()

        # v0.9 stored this as an Nx1 array instead of a flat 1-d array
        if len(dxcm.shape) == 2:
            dxcm = dxcm[:, 0]

        dxcm, dxtime = align_running_speed(dxcm, dxtime, timestamps)

        return dxcm, dxtime

    def get_pupil_location(self, as_spherical=True):
        '''Returns the x, y pupil location.

        Parameters
        ----------
        as_spherical : bool
            Whether to return the location as spherical (default) or
            not. If true, the result is altitude and azimuth in
            degrees, otherwise it is x, y in centimeters. (0,0) is
            the center of the monitor.

        Returns
        -------
        (timestamps, location)
            Timestamps is an (Nx1) array of timestamps in seconds.
            Location is an (Nx2) array of spatial location.
        '''
        if as_spherical:
            location_key = "pupil_location_spherical"
        else:
            location_key = "pupil_location"
        try:
            with h5py.File(self.nwb_file, 'r') as f:
                eye_tracking = f['processing'][self.PIPELINE_DATASET][
                    'EyeTracking'][location_key]
                pupil_location = eye_tracking['data'].value
                pupil_times = eye_tracking['timestamps'].value
        except KeyError:
            raise NoEyeTrackingException("No eye tracking for this experiment.")

        return pupil_times, pupil_location

    def get_pupil_size(self):
        '''Returns the pupil area in pixels.

        Returns
        -------
        (timestamps, areas)
            Timestamps is an (Nx1) array of timestamps in seconds.
            Areas is an (Nx1) array of pupil areas in pixels.
        '''
        try:
            with h5py.File(self.nwb_file, 'r') as f:
                pupil_tracking = f['processing'][self.PIPELINE_DATASET][
                    'PupilTracking']['pupil_size']
                pupil_size = pupil_tracking['data'].value
                pupil_times = pupil_tracking['timestamps'].value
        except KeyError:
            raise NoEyeTrackingException("No pupil tracking for this experiment.")

        return pupil_times, pupil_size

    def get_motion_correction(self):
        ''' Returns a Panda DataFrame containing the x- and y- translation of each image used for image alignment
        '''
        
        motion_correction = None
        with h5py.File(self.nwb_file, 'r') as f:
            pipeline_ds = f['processing'][self.PIPELINE_DATASET]

            # pipeline 0.9 stores this in xy_translations
            # pipeline 1.0 stores this in xy_translation
            for mc_ds_name in self.MOTION_CORRECTION_DATASETS:
                try:
                    mc_ds = pipeline_ds[mc_ds_name]

                    motion_log = mc_ds['data'].value
                    motion_time = mc_ds['timestamps'].value
                    motion_names = mc_ds['feature_description'].value

                    motion_correction = pd.DataFrame(motion_log, columns=motion_names)
                    motion_correction['timestamp'] = motion_time

                    # break out if we found it
                    break
                except KeyError as e:
                    pass
        
        if motion_correction is None:
            raise KeyError("Could not find motion correction data.")

        return motion_correction

    def save_analysis_dataframes(self, *tables):
        store = pd.HDFStore(self.nwb_file, mode='a')
        for k, v in tables:
            store.put('analysis/%s' % (k), v)
        store.close()

    def save_analysis_arrays(self, *datasets):
        with h5py.File(self.nwb_file, 'a') as f:
            for k, v in datasets:
                if k in f['analysis']:
                    del f['analysis'][k]
                f.create_dataset('analysis/%s' % k, data=v)

def align_running_speed(dxcm, dxtime, timestamps):
    ''' If running speed timestamps differ from fluorescence
    timestamps, adjust by inserting NaNs to running speed.

    Returns
    -------
    tuple: dxcm, dxtime
    '''
    if dxtime[0] != timestamps[0]:
        adjust = np.where(timestamps == dxtime[0])[0][0]
        dxtime = np.insert(dxtime, 0, timestamps[:adjust])
        dxcm = np.insert(dxcm, 0, np.repeat(np.NaN, adjust))
    adjust = len(timestamps) - len(dxtime)
    if adjust > 0:
        dxtime = np.append(dxtime, timestamps[(-1 * adjust):])
        dxcm = np.append(dxcm, np.repeat(np.NaN, adjust))
        
    return dxcm, dxtime

def warp_stimulus_coords(vertices,
                         distance=15.0,
                         mon_height_cm=32.5,
                         mon_width_cm=51.0,
                         mon_res=(1920, 1200),
                         eyepoint=(0.5, 0.5)):
    '''
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

    '''

    mon_width_cm = float(mon_width_cm)
    mon_height_cm = float(mon_height_cm)
    distance = float(distance)
    mon_res_x, mon_res_y = float(mon_res[0]), float(mon_res[1])

    vertices = vertices.astype(np.float)

    # from pixels (-1920/2 -> 1920/2) to stimulus space (-0.5->0.5)
    vertices[:, 0] = vertices[:, 0] / mon_res_x
    vertices[:, 1] = vertices[:, 1] / mon_res_y

    x = (vertices[:, 0] + 0.5) * mon_width_cm
    y = (vertices[:, 1] + 0.5) * mon_height_cm

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


def make_display_mask(display_shape=(1920, 1200)):
    ''' Build a display-shaped mask that indicates which pixels are on screen after warping the stimulus. '''
    x = np.array(range(display_shape[0])) - display_shape[0] / 2
    y = np.array(range(display_shape[1])) - display_shape[1] / 2
    display_coords = np.array(list(itertools.product(x, y)))

    warped_coords = warp_stimulus_coords(display_coords).astype(int)

    off_warped_coords = np.array([warped_coords[:, 0] + display_shape[0] / 2,
                                  warped_coords[:, 1] + display_shape[1] / 2])

    used_coords = set()
    for i in range(off_warped_coords.shape[1]):
        used_coords.add((off_warped_coords[0, i], off_warped_coords[1, i]))

    used_coords = (np.array([x for (x, y) in used_coords]).astype(int),
                   np.array([y for (x, y) in used_coords]).astype(int))

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
            tdcm = np.where((template_display_coords[0, :, :] == x) & (
                template_display_coords[1, :, :] == y))
            v = display_mask[tdcm]
            f = np.sum(v) / len(v)
            frac[x, y] = f
            mask[x, y] = f >= threshold

    return mask, frac


def _get_abstract_feature_series_stimulus_table(nwb_file, stimulus_name):
    ''' Return the a stimulus table for an abstract feature series.

    Returns
    -------
    stimulus table: pd.DataFrame
    '''


    k = "stimulus/presentation/%s" % stimulus_name

    with h5py.File(nwb_file, 'r') as f:
        if k not in f:
            raise MissingStimulusException(
                "Stimulus not found: %s" % stimulus_name)
        stim_data = f[k + '/data'].value
        features = [ v.decode('UTF-8') for v in f[k + '/features'].value ]
        frame_dur = f[k + '/frame_duration'].value

    stimulus_table = pd.DataFrame(stim_data, columns=features)
    stimulus_table.loc[:, 'start'] = frame_dur[:, 0].astype(int)
    stimulus_table.loc[:, 'end'] = frame_dur[:, 1].astype(int)

    return stimulus_table


def _get_indexed_time_series_stimulus_table(nwb_file, stimulus_name):
    ''' Return the a stimulus table for an indexed time series.

    Returns
    -------
    stimulus table: pd.DataFrame
    '''

    k = "stimulus/presentation/%s" % stimulus_name

    with h5py.File(nwb_file, 'r') as f:
        if k not in f:
            raise MissingStimulusException(
                "Stimulus not found: %s" % stimulus_name)
        inds = f[k + '/data'].value
        frame_dur = f[k + '/frame_duration'].value

    stimulus_table = pd.DataFrame(inds, columns=['frame'])
    stimulus_table.loc[:, 'start'] = frame_dur[:, 0].astype(int)
    stimulus_table.loc[:, 'end'] = frame_dur[:, 1].astype(int)

    return stimulus_table

