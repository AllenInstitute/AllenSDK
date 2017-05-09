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

import scipy.stats as st
import pandas as pd
import numpy as np
import h5py
from .stimulus_analysis import StimulusAnalysis
from .brain_observatory_exceptions import MissingStimulusException
from . import stimulus_info as stiminfo
from . import circle_plots as cplots

class NaturalMovie(StimulusAnalysis):
    """ Perform tuning analysis specific to natural movie stimulus.

    Parameters
    ----------
    data_set: BrainObservatoryNwbDataSet object

    movie_name: string
        one of [ stimulus_info.NATURAL_MOVIE_ONE, stimulus_info.NATURAL_MOVIE_TWO,
                 stimulus_info.NATURAL_MOVIE_THREE ]
    """

    def __init__(self, data_set, movie_name, **kwargs):
        super(NaturalMovie, self).__init__(data_set, **kwargs)
        
        self.movie_name = movie_name
        self._sweeplength = NaturalMovie._PRELOAD
        self._sweep_response = NaturalMovie._PRELOAD

    @property
    def sweeplength(self):
        if self._sweeplength is NaturalMovie._PRELOAD:
            self.populate_stimulus_table()

        return self._sweeplength

    @property
    def sweep_response(self):
        if self._sweep_response is NaturalMovie._PRELOAD:
            self._sweep_response = self.get_sweep_response()

        return self._sweep_response

    def populate_stimulus_table(self):
        stimulus_table = self.data_set.get_stimulus_table(self.movie_name)
        self._stim_table = stimulus_table[stimulus_table.frame == 0]
        self._sweeplength = \
            self.stim_table.start.iloc[1] - self.stim_table.start.iloc[0]

    def get_sweep_response(self):
        ''' Returns the dF/F response for each cell

        Returns
        -------
        Numpy array
        '''
        sweep_response = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(
            range(self.numbercells)).astype(str))
        for index, row in self.stim_table.iterrows():
            start = row.start
            end = start + self.sweeplength
            for nc in range(self.numbercells):
                sweep_response[str(nc)][index] = self.dfftraces[nc, start:end]
        return sweep_response

    def get_peak(self):
        ''' Computes properties of the peak response condition for each cell.

        Returns
        -------
        Pandas data frame with the below fields. A suffix of "nm1", "nm2" or "nm3" is appended to the field name depending
        on which of three movie clips was presented.
            * peak_nm1 (frame with peak response)
            * response_variability_nm1
        '''
        peak_movie = pd.DataFrame(index=range(self.numbercells), columns=(
            'peak', 'response_reliability', 'cell_specimen_id'))
        cids = self.data_set.get_cell_specimen_ids()

        mask = np.ones((10,10))
        for i in range(10):
            for j in range(10):
                if i>=j:
                    mask[i,j] = np.NaN        
        
        for nc in range(self.numbercells):
            peak_movie.cell_specimen_id.iloc[nc] = cids[nc]
            meanresponse = self.sweep_response[str(nc)].mean()
            
#            movie_len = len(meanresponse) / 30
#            output = np.empty((movie_len, 10))
#            for tr in range(10):
#                test = self.sweep_response[str(nc)].iloc[tr]
#                for i in range(movie_len):
#                    _, p = st.ks_2samp(
#                        test[i * 30:(i + 1) * 30], test[(i + 1) * 30:(i + 2) * 30])
#                    output[i, tr] = p
#            output = np.where(output < 0.05, 1, 0)
#            ptime = np.sum(output, axis=1)
#            ptime *= 10
            peak = np.argmax(meanresponse)
#            if peak > 30:
#                peak_movie.response_reliability.iloc[
#                    nc] = ptime[(peak - 30) / 30]
#            else:
#                peak_movie.response_reliability.iloc[nc] = ptime[0]
            peak_movie.peak.iloc[nc] = peak
            
            #reliability
            corr_matrix = np.empty((10,10))
            for i in range(10):
                for j in range(10):
                    r,p = st.pearsonr(self.sweep_response[str(nc)].iloc[i], self.sweep_response[str(nc)].iloc[j])
                    corr_matrix[i,j] = r
            corr_matrix*=mask
            peak_movie.response_reliability.iloc[nc] = np.nanmean(corr_matrix)
            
        if self.movie_name == stiminfo.NATURAL_MOVIE_ONE:
            peak_movie.rename(columns={
                              'peak': 'peak_'+stiminfo.NATURAL_MOVIE_ONE_SHORT, 
                              'response_reliability': 'response_reliability_'+stiminfo.NATURAL_MOVIE_ONE_SHORT}, 
                              inplace=True)
        elif self.movie_name == stiminfo.NATURAL_MOVIE_TWO:
            peak_movie.rename(columns={
                              'peak': 'peak_'+stiminfo.NATURAL_MOVIE_TWO_SHORT, 
                              'response_reliability': 'response_reliability_'+stiminfo.NATURAL_MOVIE_TWO_SHORT},
                              inplace=True)
        elif self.movie_name == stiminfo.NATURAL_MOVIE_THREE:
            peak_movie.rename(columns={
                              'peak': 'peak_'+stiminfo.NATURAL_MOVIE_THREE_SHORT, 
                              'response_reliability': 'response_reliability_'+stiminfo.NATURAL_MOVIE_THREE_SHORT}, 
                              inplace=True)

        return peak_movie

    def open_track_plot(self, cell_specimen_id=None, cell_index=None):
        cell_index = self.row_from_cell_id(cell_specimen_id, cell_index)

        cell_rows = self.sweep_response[str(cell_index)]
        data = []
        for i in range(len(cell_rows)):
            data.append(cell_rows.iloc[i])

        data = np.vstack(data)

        tp = cplots.TrackPlotter(ring_length=360)
        tp.plot(data,
                clim=[0, data.mean() + data.std()*3])
        tp.show_arrow()

    @staticmethod 
    def from_analysis_file(data_set, analysis_file, movie_name):
        nm = NaturalMovie(data_set, movie_name)
        nm.populate_stimulus_table()

        # TODO: deal with this properly
        suffix_map = {
            stiminfo.NATURAL_MOVIE_ONE: '_'+stiminfo.NATURAL_MOVIE_ONE_SHORT,
            stiminfo.NATURAL_MOVIE_TWO: '_'+stiminfo.NATURAL_MOVIE_TWO_SHORT,
            stiminfo.NATURAL_MOVIE_THREE: '_'+stiminfo.NATURAL_MOVIE_THREE_SHORT
            }

        try:
            suffix = suffix_map[movie_name]


            nm._sweep_response = pd.read_hdf(analysis_file, "analysis/sweep_response"+suffix)
            nm._peak = pd.read_hdf(analysis_file, "analysis/peak")

            with h5py.File(analysis_file, "r") as f:
                nm._binned_dx_sp = f["analysis/binned_dx_sp"].value
                nm._binned_cells_sp = f["analysis/binned_cells_sp"].value
                nm._binned_dx_vis = f["analysis/binned_dx_vis"].value
                nm._binned_cells_vis = f["analysis/binned_cells_vis"].value
        except Exception as e:
            raise MissingStimulusException(e.args)

        return nm
