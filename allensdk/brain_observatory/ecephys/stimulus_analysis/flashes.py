import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit

from .stimulus_analysis import StimulusAnalysis, get_fr


class Flashes(StimulusAnalysis):
    """
    A class for computing single-unit metrics from the full-field flash stimulus of an ecephys session NWB file.

    To use, pass in a EcephysSession object::
        session = EcephysSession.from_nwb_path('/path/to/my.nwb')
        fl_analysis = Flashes(session)

    or, alternatively, pass in the file path::
        fl_analysis = Flashes('/path/to/my.nwb')

    You can also pass in a unit filter dictionary which will only select units with certain properties. For example
    to get only those units which are on probe C and found in the VISp area::
        fl_analysis = Flashes(session, filter={'location': 'probeC', 'structure_acronym': 'VISp'})

    To get a table of the individual unit metrics ranked by unit ID::
        metrics_table_df = fl_analysis.metrics()

    """

    def __init__(self, ecephys_session, **kwargs):
        super(Flashes, self).__init__(ecephys_session, **kwargs)

        self._metrics = None

        self._colors = None
        self._col_color = 'Color'

        if self._params is not None:
            self._params = self._params['flashes']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'flash_250ms'

        self._trial_duration = 0.25

        self._module_name = 'Flashes'


    @property
    def colors(self):
        """ Array of 'color' conditions (black vs. white flash) """
        if self._color is None:
            self._get_stim_table_stats()

        return self._colors
    

    @property
    def null_condition(self):
        """ Stimulus condition ID for null stimulus (not used, so set to -1) """
        return -1
    
    @property
    def METRICS_COLUMNS(self):
        return [('on_off_ratio_fl', np.float64), 
                ('sustained_idx_fl', np.float64),
                ('firing_rate_fl', np.float64), 
                ('reliability_fl', np.float64),
                ('fano_fl', np.float64), 
                ('lifetime_sparseness_fl', np.float64), 
                ('run_pval_fl', np.float64),
                ('run_mod_fl', np.float64)]

    @property
    def metrics(self):

        if self._metrics is None:

            print('Calculating metrics for ' + self.name)

            unit_ids = self.unit_ids
        
            metrics_df = self.empty_metrics_table()

            metrics_df['on_off_ratio_fl'] = [self._get_on_off_ratio(unit) for unit in unit_ids]
            metrics_df['sustained_idx_fl'] = [self._get_sustained_index(unit) for unit in unit_ids]
            metrics_df['firing_rate_fl'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['reliability_fl'] = [self.get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['fano_fl'] = [self.get_fano_factor(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_fl'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_fl', 'run_mod_fl']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        """ Extract colors from the stimulus table """

        self._colors = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_color] != 'null'][self._col_color].unique())


    def _get_sustained_index(self, unit_id):

        """ Calculate the sustained index for a given unit, a measure of the transience of
        the flash response.

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        sustained_index - metric

        """

        return np.nan

    def _get_on_off_ratio(self, unit_id):

        """ Calculate the ratio of the on response vs. off response for a given unit

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        on_off_ratio - metric

        """

        return np.nan

