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

import numpy as np
from .stimulus_analysis import StimulusAnalysis
import stimulus_info
import scipy.ndimage 

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

    def __init__(self, data_set, stimulus, **kwargs):
        super(LocallySparseNoise, self).__init__(data_set, **kwargs)
        self.stimulus = stimulus
        
        try:
            lsn_dims = stimulus_info.LOCALLY_SPARSE_NOISE_DIMENSIONS[self.stimulus]
        except KeyError as e:
            raise KeyError("Unknown stimulus name: %s" % self.stimulus)
        
        self.nrows = lsn_dims[0]
        self.ncols = lsn_dims[1]
        
        self._stim_table = StimulusAnalysis._PRELOAD
        self._LSN = StimulusAnalysis._PRELOAD
        self._LSN_mask = StimulusAnalysis._PRELOAD
        self._sweeplength = StimulusAnalysis._PRELOAD
        self._interlength = StimulusAnalysis._PRELOAD
        self._extralength = StimulusAnalysis._PRELOAD

        self._sweep_response = StimulusAnalysis._PRELOAD
        self._mean_sweep_response = StimulusAnalysis._PRELOAD
        self._pval = StimulusAnalysis._PRELOAD\

        self._receptive_field = StimulusAnalysis._PRELOAD

        # get stimulus table

    @property
    def stim_table(self):
        if self._stim_table is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._stim_table

    @property
    def LSN(self):
        if self._LSN is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._LSN

    @property
    def LSN_mask(self):
        if self._LSN_mask is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._LSN_mask

    @property
    def sweeplength(self):
        if self._sweeplength is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._sweeplength

    @property
    def interlength(self):
        if self._interlength is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._interlength

    @property
    def extralength(self):
        if self._extralength is StimulusAnalysis._PRELOAD:
            self.populate_stimulus_table()

        return self._extralength

    @property
    def sweep_response(self):
        if self._sweep_response is StimulusAnalysis._PRELOAD:
            self._sweep_response, self._mean_sweep_response, self._pval = \
                self.get_sweep_response()

        return self._sweep_response

    @property
    def mean_sweep_response(self):
        if self._mean_sweep_response is StimulusAnalysis._PRELOAD:
            self._sweep_response, self._mean_sweep_response, self._pval = \
                self.get_sweep_response()

        return self._mean_sweep_response

    @property
    def pval(self):
        if self._pval is StimulusAnalysis._PRELOAD:
            self._sweep_response, self._mean_sweep_response, self._pval = \
                self.get_sweep_response()

        return self._pval

    @property
    def receptive_field(self):
        if self._receptive_field is StimulusAnalysis._PRELOAD:
            self._receptive_field = self.get_receptive_field()

        return self._receptive_field

    def populate_stimulus_table(self):
        self._stim_table = self.data_set.get_stimulus_table(self.stimulus)
        self._LSN, self._LSN_mask = self.data_set.get_locally_sparse_noise_stimulus_template(
            self.stimulus, mask_off_screen=False)
        self._sweeplength = self._stim_table['end'][
            1] - self._stim_table['start'][1]
        self._interlength = 4 * self._sweeplength
        self._extralength = self._sweeplength

    def get_receptive_field(self):
        ''' Calculates receptive fields for each cell
        '''
        print("Calculating mean responses")
        receptive_field = np.empty(
            (self.nrows, self.ncols, self.numbercells + 1, 2))

        for xp in range(self.nrows):
            for yp in range(self.ncols):
                on_frame = np.where(self.LSN[:, xp, yp] == self.LSN_ON)[0]
                off_frame = np.where(self.LSN[:, xp, yp] == self.LSN_OFF)[0]
                subset_on = self.mean_sweep_response[
                    self.stim_table.frame.isin(on_frame)]
                subset_off = self.mean_sweep_response[
                    self.stim_table.frame.isin(off_frame)]
                receptive_field[xp, yp, :, 0] = subset_on.mean(axis=0)
                receptive_field[xp, yp, :, 1] = subset_off.mean(axis=0)
        return receptive_field

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
        

        
