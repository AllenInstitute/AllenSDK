import numpy as np
import pandas as pd
from six import string_types
import scipy.stats as st
import logging
import matplotlib.pyplot as plt

from .stimulus_analysis import StimulusAnalysis

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logger = logging.getLogger(__name__)


class NaturalScenes(StimulusAnalysis):
    """
    A class for computing single-unit metrics from the natural scenes stimulus of an ecephys session NWB file.

    To use, pass in a EcephysSession object::
        session = EcephysSession.from_nwb_path('/path/to/my.nwb')
        ns_analysis = NaturalScenes(session)

    or, alternatively, pass in the file path::
        ns_analysis = NaturalScenes('/path/to/my.nwb')

    You can also pass in a unit filter dictionary which will only select units with certain properties. For example
    to get only those units which are on probe C and found in the VISp area::
        ns_analysis = NaturalScenes(session, filter={'location': 'probeC', 'structure_acronym': 'VISp'})

    To get a table of the individual unit metrics ranked by unit ID::
        metrics_table_df = ns_analysis.metrics()

    """

    def __init__(self, ecephys_session, col_image='Image', trial_duration=0.25, **kwargs):
        super(NaturalScenes, self).__init__(ecephys_session, trial_duration=trial_duration, **kwargs)

        self._images = None
        self._number_images = None
        self._number_nonblank = None  # does not include Image number = -1.
        self._mean_sweep_events = None
        self._response_events = None
        self._response_trials = None
        self._metrics = None

        self._col_image = col_image # 'Image'

        # self._trial_duration = 0.25  # Passed in to kwargs and read by parent

        if self._params is not None:
            self._params = self._params.get('natural_scenes', {})
            self._stimulus_key = self._params.get('stimulus_key', None)  # Overwrites parent value with argvars
        else:
            self._params = {}

        self._module_name = 'Natural Scenes'

    @property
    def images(self):
        """ Array of iamge labels """
        if self._images is None:
            self._get_stim_table_stats()

        return self._images

    @property
    def frames(self):
        # Required to deal with naming difference between NWB 1 and 2
        return self.images

    @property
    def number_images(self):
        """ Number of images shown """
        if self._images is None:
            self._get_stim_table_stats()

        return self._number_images

    @property
    def number_nonblank(self):
        """ Number of images shown (excluding blank condition) """
        if self._number_nonblank is None:
            self._get_stim_table_stats()

        return self._number_nonblank

    @property
    def null_condition(self):
        """ Stimulus condition ID for null (blank) stimulus """
        return self.stimulus_conditions[self.stimulus_conditions[self._col_image] == -1].index

    @property
    def METRICS_COLUMNS(self):
        return [('pref_image_ns', np.uint64), 
                ('image_selectivity_ns', np.float64), 
                ('firing_rate_ns', np.float64), 
                ('fano_ns', np.float64),
                ('time_to_peak_ns', np.float64), 
                ('reliability_ns', np.float64),
                ('lifetime_sparseness_ns', np.float64), 
                ('run_pval_ns', np.float64), 
                ('run_mod_ns', np.float64)]

    @property
    def metrics(self):

        if self._metrics is None:
            logger.info('Calculating metrics for ' + self.name)

            unit_ids = self.unit_ids

            metrics_df = self.empty_metrics_table()
            metrics_df['pref_image_ns'] = [self.get_preferred_condition(unit) for unit in unit_ids]
            metrics_df['image_selectivity_ns'] = [self._get_image_selectivity(unit) for unit in unit_ids]
            metrics_df['firing_rate_ns'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['fano_ns'] = [self.get_fano_factor(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['time_to_peak_ns'] = [self.get_time_to_peak(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['reliability_ns'] = [self.get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_ns'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_ns', 'run_mod_ns']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics

    @property
    def known_stimulus_keys(self):
        return ['natural_scenes', 'Natural_Images', 'Natural Images']


    def _get_stim_table_stats(self):

        """ Extract image labels from the stimulus table """

        self._images = np.sort(self.stimulus_conditions[self._col_image].unique()).astype(np.int64)
        self._number_images = len(self._images)
        self._number_nonblank = len(self._images[self._images >= 0])


    def _get_image_selectivity(self, unit_id, num_steps=1000):
        """ Calculate the image selectivity for a given unit

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        image_selectivity - metric

        """

        unit_stats = self.conditionwise_statistics.loc[unit_id].drop(index=self.null_condition)

        fmin = unit_stats['spike_mean'].min()
        fmax = unit_stats['spike_mean'].max()

        j = np.arange(num_steps)
        thresh = fmin + j * ((fmax-fmin) / len(j))
        rtj = [np.mean(unit_stats['spike_mean'] > t) for t in thresh]

        return 1 - (2*np.mean(rtj))
