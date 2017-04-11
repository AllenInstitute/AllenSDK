# Copyright 2016-2017 Allen Institute for Brain Science
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

import allensdk.brain_observatory.stimulus_info as stimulus_info
import h5py
import numpy as np
import pandas as pd
import scipy.ndimage
from .receptive_field_analysis import receptive_field_analysis

from . import circle_plots as cplots
from . import observatory_plots as oplots
from .brain_observatory_exceptions import MissingStimulusException
from .stimulus_analysis import StimulusAnalysis
from .receptive_field_analysis.tools import dict_generator, read_h5_group

class LocallySparseNoise(StimulusAnalysis):
    """ Perform tuning analysis specific to the locally sparse noise stimulus.

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object

    stimulus: string
       Name of locally sparse noise stimulus.  See brain_observatory.stimulus_info.

    nrows: int
       Number of rows in the stimulus template

    ncol: int
       Number of columns in the stimulus template
    """

    LSN_ON = 255
    LSN_OFF = 0
    LSN_GREY = 127
    LSN_OFF_SCREEN = 64

    def __init__(self, data_set, stimulus=None, **kwargs):
        super(LocallySparseNoise, self).__init__(data_set, **kwargs)
        if stimulus is None:
            self.stimulus = stimulus_info.LOCALLY_SPARSE_NOISE
        else:
            self.stimulus = stimulus

        try:
            lsn_dims = stimulus_info.LOCALLY_SPARSE_NOISE_DIMENSIONS[self.stimulus]
        except KeyError as e:
            raise KeyError("Unknown stimulus name: %s" % self.stimulus)
        
        self.nrows = lsn_dims[0]
        self.ncols = lsn_dims[1]
        
        self._LSN = LocallySparseNoise._PRELOAD
        self._LSN_mask = LocallySparseNoise._PRELOAD
        self._sweeplength = LocallySparseNoise._PRELOAD
        self._interlength = LocallySparseNoise._PRELOAD
        self._extralength = LocallySparseNoise._PRELOAD
        self._mean_response = LocallySparseNoise._PRELOAD
        self._receptive_field = LocallySparseNoise._PRELOAD
        self._csid_receptive_field_analysis_data_dict = LocallySparseNoise._PRELOAD

    @property
    def LSN(self):
        if self._LSN is LocallySparseNoise._PRELOAD:
            self.populate_stimulus_table()

        return self._LSN

    @property
    def LSN_mask(self):
        if self._LSN_mask is LocallySparseNoise._PRELOAD:
            self.populate_stimulus_table()

        return self._LSN_mask

    @property
    def sweeplength(self):
        if self._sweeplength is LocallySparseNoise._PRELOAD:
            self.populate_stimulus_table()

        return self._sweeplength

    @property
    def interlength(self):
        if self._interlength is LocallySparseNoise._PRELOAD:
            self.populate_stimulus_table()

        return self._interlength

    @property
    def extralength(self):
        if self._extralength is LocallySparseNoise._PRELOAD:
            self.populate_stimulus_table()

        return self._extralength

    @property
    def receptive_field(self):
        if self._receptive_field is LocallySparseNoise._PRELOAD:
            self._receptive_field = self.get_receptive_field()

        return self._receptive_field

    @property
    def csid_receptive_field_analysis_data_dict(self):
        if self._csid_receptive_field_analysis_data_dict is LocallySparseNoise._PRELOAD:
            self._csid_receptive_field_analysis_data_dict = self.get_receptive_field_analysis_data()

        return self._csid_receptive_field_analysis_data_dict

    @property
    def mean_response(self):
        if self._mean_response is LocallySparseNoise._PRELOAD:
            self._mean_response = self.get_mean_response()

        return self._mean_response

    def populate_stimulus_table(self):
        self._stim_table = self.data_set.get_stimulus_table(self.stimulus)
        self._LSN, self._LSN_mask = self.data_set.get_locally_sparse_noise_stimulus_template(
            self.stimulus, mask_off_screen=False)
        self._sweeplength = self._stim_table['end'][
            1] - self._stim_table['start'][1]
        self._interlength = 4 * self._sweeplength
        self._extralength = self._sweeplength

    def get_mean_response(self):

        print("Calculating mean responses")
        mean_response = np.empty(
            (self.nrows, self.ncols, self.numbercells + 1, 2))

        for xp in range(self.nrows):
            for yp in range(self.ncols):
                on_frame = np.where(self.LSN[:, xp, yp] == self.LSN_ON)[0]
                off_frame = np.where(self.LSN[:, xp, yp] == self.LSN_OFF)[0]
                subset_on = self.mean_sweep_response[
                    self.stim_table.frame.isin(on_frame)]
                subset_off = self.mean_sweep_response[
                    self.stim_table.frame.isin(off_frame)]
                mean_response[xp, yp, :, 0] = subset_on.mean(axis=0)
                mean_response[xp, yp, :, 1] = subset_off.mean(axis=0)
        return mean_response

    def get_receptive_field(self):
        ''' Calculates receptive fields for each cell
        '''

        receptive_field = np.zeros((self.nrows, self.ncols, self.numbercells, 2))
        csid_list = self.data_set.get_cell_specimen_ids()
        import warnings
        warnings.warn('comment in this line when done debugging')
        # assert len(csid_list) == len(self.csid_receptive_field_analysis_data_dict)


        warnings.warn('change this from 2')
        for csid in csid_list[:2]:
            cell_index = self.data_set.get_cell_specimen_indices(cell_specimen_ids=[csid])[0]
            curr_receptive_field_data_dict = self.csid_receptive_field_analysis_data_dict[str(csid)]
            rf_on = curr_receptive_field_data_dict['on']['rts_convolution']['data'].copy()
            rf_off = curr_receptive_field_data_dict['off']['rts_convolution']['data'].copy()
            rf_on[np.logical_not(curr_receptive_field_data_dict['on']['fdr_mask']['data'].sum(axis=0))] = np.nan
            rf_off[np.logical_not(curr_receptive_field_data_dict['off']['fdr_mask']['data'].sum(axis=0))] = np.nan
            receptive_field[:,:,cell_index, 0] = rf_on
            receptive_field[:, :, cell_index, 1] = rf_off

        return receptive_field


    def get_receptive_field_analysis_data(self):
        ''' Calculates receptive fields for each cell
        '''

        print("Calculating receptive analysis")
        csid_receptive_field_analysis_data_dict = receptive_field_analysis(self)
        return csid_receptive_field_analysis_data_dict



    @staticmethod
    def merge_receptive_fields(rc1, rc2):
        """ TODO
        """

        # make sure that rc1 is the larger one
        if rc2.shape[0] > rc1.shape[0]:
            rc1, rc2 = rc2, rc1

        shape_mult = np.array(rc1.shape) / np.array(rc2.shape)

        rc2_zoom = scipy.ndimage.zoom(rc2, shape_mult, order=0)

        return rc1 + rc2_zoom

    def plot_receptive_field(self, cell_specimen_id, on, color_map=None, zlim=[-3,3], mask=None):
        csids = self.data_set.get_cell_specimen_ids()
        cell_idx = np.where(csids == cell_specimen_id)[0][0]
        
        if color_map is None:
            color_map = oplots.LSN_RF_ON_COLOR_MAP if on else oplots.LSN_RF_OFF_COLOR_MAP

        rf_idx = 0 if on else 1
        cell_rf = self.receptive_field[:,:,cell_idx,rf_idx]
        
        oplots.plot_receptive_field(cell_rf, color_map, zlim, mask)

    def sort_trials(self):
        ds = self.data_set

        lsn_movie, lsn_mask = ds.get_locally_sparse_noise_stimulus_template(self.stimulus, 
                                                                            mask_off_screen=False)

        baseline_trials = np.unique(np.where(lsn_movie[:,-5:,-1] != LocallySparseNoise.LSN_GREY)[0])
        baseline_df = self.mean_sweep_response.ix[baseline_trials]
        cell_baselines = np.nanmean(baseline_df.values, axis=0)

        lsn_movie[:,~lsn_mask] = LocallySparseNoise.LSN_OFF_SCREEN

        trials = {}
        for row in range(self.nrows):
            for col in range(self.ncols):
                on_trials = np.where(lsn_movie[:,row,col] == LocallySparseNoise.LSN_ON)
                off_trials = np.where(lsn_movie[:,row,col] == LocallySparseNoise.LSN_OFF)

                trials[(col,row,True)] = on_trials
                trials[(col,row,False)] = off_trials

        return trials, cell_baselines


    def open_pincushion_plot(self, cell_specimen_id, on, color_map=None):
        csids = self.data_set.get_cell_specimen_ids()
        cell_id = np.where(csids == cell_specimen_id)[0][0]

        trials, baselines = self.sort_trials()
        data = self.mean_sweep_response[str(cell_id)].values
        
        cplots.make_pincushion_plot(data, trials, on, 
                                    self.nrows, self.ncols,
                                    clim=[ baselines[cell_id], data.mean() + data.std() * 3 ],
                                    color_map=color_map,
                                    radius=1.0/16.0)

    @staticmethod
    def from_analysis_file(data_set, analysis_file, stimulus):
        lsn = LocallySparseNoise(data_set, stimulus)
        lsn.populate_stimulus_table()

        try:

            # import warnings
            # warnings.warn('COmment this back in')
            lsn._sweep_response = pd.read_hdf(analysis_file, "analysis/sweep_response_lsn")
            lsn._mean_sweep_response = pd.read_hdf(analysis_file, "analysis/mean_sweep_response_lsn")

            with h5py.File(analysis_file, "r") as f:
                lsn._mean_response = f["analysis/mean_response_lsn"].value

            with h5py.File(analysis_file, "r") as f:
                lsn._csid_receptive_field_analysis_data_dict = LocallySparseNoise.read_csid_receptive_field_analysis_dict(f)

        except Exception as e:
            raise MissingStimulusException(e.args)

        return lsn


    def save_csid_receptive_field_analysis_dict(self):

        prefix = 'receptive_field_analysis'

        attr_list = []
        file_handle = h5py.File(self.data_set.nwb_file, 'a')
        if prefix in file_handle['analysis']:
            del file_handle['analysis'][prefix]
        f = file_handle.create_group('analysis/%s' % prefix)
        for x in dict_generator(self.csid_receptive_field_analysis_data_dict):
            if x[-2] == 'data':
                f['/'.join(x[:-1])] = x[-1]
            elif x[-3] == 'attrs':
                attr_list.append(x)
            else:
                raise Exception

        for x in attr_list:
            if len(x) > 3:
                f['/'.join(x[:-3])].attrs[x[-2]] = x[-1]
            else:
                assert len(x) == 3

                if x[-1] is None:
                    f.attrs[x[-2]] = np.NaN
                else:
                    f.attrs[x[-2]] = x[-1]

        file_handle.close()
        

    @staticmethod
    def read_csid_receptive_field_analysis_dict(file_handle, path=None):

        prefix = 'receptive_field_analysis'

        f = file_handle['analysis/%s' % prefix]
        if path is None:
            receptive_field_data_dict = read_h5_group(f)
        else:
            receptive_field_data_dict = read_h5_group(f[path])

        return receptive_field_data_dict