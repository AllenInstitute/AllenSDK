import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis, get_fr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DotMotion(StimulusAnalysis):
    """
    A class for computing single-unit metrics from the dot motion stimulus of an ecephys session NWB file.

    To use, pass in a EcephysSession object::
        session = EcephysSession.from_nwb_path('/path/to/my.nwb')
        dm_analysis = DotMotion(session)

    or, alternatively, pass in the file path::
        dm_analysis = DotMotion('/path/to/my.nwb')

    You can also pass in a unit filter dictionary which will only select units with certain properties. For example
    to get only those units which are on probe C and found in the VISp area::
        dm_analysis = DotMotion(session, filter={'location': 'probeC', 'structure_acronym': 'VISp'})

    To get a table of the individual unit metrics ranked by unit ID::
        metrics_table_df = dm_analysis.metrics()

    """

    def __init__(self, ecephys_session, **kwargs):
        super(DotMotion, self).__init__(ecephys_session, **kwargs)

        self._dirvals = None
        self._col_dir = 'Ori'

        self._speeds = None
        self._col_speed = 'Speed'

        self._trial_duration = 1.0

        if self._params is not None:
            self._params = self._params['dot_motion']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'motion_stimulus'

        self._module_name = 'Dot Motion'

    @property
    def directions(self):
        """ Array of dot motion direction conditions """
        if self._dirvals is None:
            self._get_stim_table_stats()

        return self._dirvals
    
    @property
    def speeds(self):
        """ Array of dot motion speed conditions """
        if self._speeds is None:
            self._get_stim_table_stats()

        return self._speeds

    @property
    def null_condition(self):
        """ Stimulus condition ID for null stimulus (not used, so set to -1) """
        return -1

    @property
    def METRICS_COLUMNS(self):
        return [('pref_speed_dm', np.float64), 
                ('pref_dir_dm', np.float64), 
                ('firing_rate_dm', np.float64), 
                ('fano_dm', np.float64),
                ('speed_tuning_idx_dm', np.float64), 
                ('time_to_peak_dm', np.float64), 
                ('reliability_dm', np.float64),
                ('lifetime_sparseness_dm', np.float64), 
                ('run_mod_dm', np.float64), 
                ('run_pval_dm', np.float64)]

    @property
    def metrics(self):

        if self._metrics is None:

            print('Calculating metrics for ' + self.name)

            unit_ids = self.unit_ids
        
            metrics_df = self.empty_metrics_table()

            metrics_df['pref_speed_dm'] = [self._get_pref_speed(unit) for unit in unit_ids]
            metrics_df['pref_dir_dm'] = [self._get_pref_dir(unit) for unit in unit_ids]
            metrics_df['firing_rate_dm'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['fano_dm'] = [self.get_fano_factor(unit, get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['speed_tuning_idx_dm'] = [self._get_speed_tuning_index(unit) for unit in unit_ids]
            metrics_df['reliability_dm'] = [self.get_reliability(unit, get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_dm'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_dm', 'run_mod_dm']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        """ Extract directions and speeds from the stimulus table """

        self._dirvals = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_dir] != 'null'][self._col_dir].unique())
        self._speeds = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_speed] != 'null'][self._col_speed].unique())


    def _get_pref_speed(self, unit_id):

        """ Calculate the preferred speed condition for a given unit

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        pref_speed - stimulus speed driving the maximal response

        """

        similar_conditions = [self.stimulus_conditions.index[self.stimulus_conditions[self._col_speed] == speed].tolist() for speed in self.speeds]
        df = pd.DataFrame(index=self.speeds,
                         data = {'spike_mean' : 
                                [self.conditionwise_statistics.loc[unit_id].loc[condition_inds]['spike_mean'].mean() for condition_inds in similar_conditions]
                             }
                         ).rename_axis(self._col_speed)

        return df.idxmax().iloc[0]


    def _get_pref_dir(self, unit_id):

        """ Calculate the preferred direction condition for a given unit

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        pref_dir - stimulus direction driving the maximal response

        """

        similar_conditions = [self.stimulus_conditions.index[self.stimulus_conditions[self._col_dir] == direction].tolist() for direction in self.directions]
        df = pd.DataFrame(index=self.directions,
                         data = {'spike_mean' : 
                                [self.conditionwise_statistics.loc[unit_id].loc[condition_inds]['spike_mean'].mean() for condition_inds in similar_conditions]
                             }
                         ).rename_axis(self._col_dir)

        return df.idxmax().iloc[0]


    def _get_speed_tuning_index(self, unit_id):

        """ Calculate the speed tuning for a given unit

        SEE: https://github.com/AllenInstitute/ecephys_analysis_modules/blob/master/ecephys_analysis_modules/modules/tuning/tuning_speed.py

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        speed_tuning - degree to which the unit's responses are modulated by stimulus speed

        """


        return np.nan

