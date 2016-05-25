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

import scipy.stats as st
import pandas as pd
import numpy as np
from allensdk.brain_observatory.stimulus_analysis import StimulusAnalysis

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
        stimulus_table = self.data_set.get_stimulus_table(movie_name)
        self.stim_table = stimulus_table[stimulus_table.frame==0]
        self.sweeplength = self.stim_table.start.iloc[1] - self.stim_table.start.iloc[0]
        self.sweep_response = self.get_sweep_response()
        self.peak = self.get_peak(movie_name=movie_name)     
        
    def get_sweep_response(self):
        ''' Returns the dF/F response for each cell

        Returns
        -------
        Numpy array
        '''
        sweep_response = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
        for index, row in self.stim_table.iterrows():
            start = row.start
            end = start + self.sweeplength
            for nc in range(self.numbercells):
                sweep_response[str(nc)][index] = self.dfftraces[nc,start:end]
        return sweep_response
    
    def get_peak(self, movie_name):
        ''' Computes properties of the peak response condition for each cell.

        Parameters
        ----------
        movie_name: string
            one of [ stimulus_info.NATURAL_MOVIE_ONE, stimulus_info.NATURAL_MOVIE_TWO, stimulus_info.NATURAL_MOVIE_THREE ]

        Returns
        -------
        Pandas data frame with the below fields. A suffix of "nm1", "nm2" or "nm3" is appended to the field name depending
        on which of three movie clips was presented.
            * peak_nm1 (frame with peak response)
            * response_variability_nm1
        '''
        peak_movie = pd.DataFrame(index=range(self.numbercells), columns=('peak','response_reliability','cell_specimen_id'))
        cids = self.data_set.get_cell_specimen_ids()

        for nc in range(self.numbercells):
            peak_movie.cell_specimen_id.iloc[nc] = cids[nc]
            meanresponse = self.sweep_response[str(nc)].mean()
            movie_len= len(meanresponse)/30
            output = np.empty((movie_len,10))
            for tr in range(10):
                test = self.sweep_response[str(nc)].iloc[tr]
                for i in range(movie_len):
                    _,p = st.ks_2samp(test[i*30:(i+1)*30],test[(i+1)*30:(i+2)*30])
                    output[i,tr] = p    
            output = np.where(output<0.05, 1, 0)
            ptime = np.sum(output, axis=1)
            ptime*=10
            peak = np.argmax(meanresponse)
            if peak>30:
                peak_movie.response_reliability.iloc[nc] = ptime[(peak-30)/30]
            else:
                peak_movie.response_reliability.iloc[nc] = ptime[0]
            peak_movie.peak.iloc[nc] = peak
        if movie_name=='natural_movie_one':
            peak_movie.rename(columns={'peak':'peak_nm1','response_reliability':'response_reliability_nm1'}, inplace=True)
        elif movie_name=='natural_movie_two':
            peak_movie.rename(columns={'peak':'peak_nm2','response_reliability':'response_reliability_nm2'}, inplace=True)
        elif movie_name=='natural_movie_three':
            peak_movie.rename(columns={'peak':'peak_nm3','response_reliability':'response_reliability_nm3'}, inplace=True)

        return peak_movie
