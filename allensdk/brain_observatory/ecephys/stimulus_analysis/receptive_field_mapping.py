import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit

from .stimulus_analysis import StimulusAnalysis


class ReceptiveFieldMapping(StimulusAnalysis):
    def __init__(self, ecephys_session, **kwargs):
        super(ReceptiveFieldMapping, self).__init__(ecephys_session, **kwargs)

        self._pos_x = None
        self._pos_y = None

        self._rf_matrix = None

        self._col_pos_x = 'Pos_x'
        self._col_pos_y = 'Pos_y'

        self._trial_duration = 0.25

        if self._params is not None:
            self._params = self._params['receptive_field_mapping']
            self._stimulus_key = self._params['stimulus_key']
        else:
            self._stimulus_key = 'gabor_20_deg_250ms_0'

        self._module_name = 'Receptive Field Mapping'

    @property
    def elevations(self):
        """ Array of stimulus elevations """
        if self._pos_y is None:
            self._get_stim_table_stats()

        return self._pos_y

    @property
    def azimuths(self):
        """ Array of stimulus azimuths """
        if self._pos_x is None:
            self._get_stim_table_stats()

        return self._pos_x

    @property
    def null_condition(self):
        """ Stimulus condition ID for null stimulus (not used, so set to -1) """
        return -1

    @property
    def receptive_fields(self):
        """ Spatial receptive fields for N units (9 x 9 x N matrix of responses) """
        if self._rf_matrix is None:

            bin_edges = np.linspace(0, 0.249, 249)

            self.stim_table.loc[:, 'Pos_y'] = 8 - self.stim_table['Pos_y']

            presentationwise_response_matrix = self.ecephys_session.presentationwise_spike_counts(
                bin_edges = bin_edges,
                stimulus_presentation_ids = self.stim_table.index.values,
                unit_ids = self.unit_ids,
                )

            self._rf_matrix = self._response_by_stimulus_position(presentationwise_response_matrix, 
                                              self.stim_table)

        return self._rf_matrix
    

    @property
    def METRICS_COLUMNS(self):
        return [('azimuth_rf', np.float64), 
                ('elevation_rf', np.float64), 
                ('width_rf', np.float64), 
                ('height_rf', np.float64),
                ('area_rf', np.float64), 
                ('exists_rf', bool), 
                ('on_screen_rf', bool), 
                ('firing_rate_rf', np.float64),
                ('fano_rf', np.float64), 
                ('time_to_peak_rf', np.float64), 
                ('reliability_rf', np.float64),
                ('lifetime_sparseness_rf', np.float64),
                ('run_mod_rf', np.float64), 
                ('run_pval_rf', np.float64)
                ]

    @property
    def metrics(self):

        if self._metrics is None:

            print('Calculating metrics for ' + self.name)

            unit_ids = self.unit_ids
        
            metrics_df = self.empty_metrics_table()

            metrics_df.loc[:, ['azimuth_rf',
                               'elevation_rf',
                               'width_rf',
                               'height_rf',
                               'area_rf',
                               'exists_rf',
                               'on_screen_rf']] = [self._get_rf_stats(unit) for unit in unit_ids]
            metrics_df['firing_rate_rf'] = [self.get_overall_firing_rate(unit) for unit in unit_ids]
            metrics_df['fano_rf'] = [self.get_fano_factor(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['time_to_peak_rf'] = [self.get_time_to_peak(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['reliability_rf'] = [self.get_reliability(unit, self.get_preferred_condition(unit)) for unit in unit_ids]
            metrics_df['lifetime_sparseness_rf'] = [self.get_lifetime_sparseness(unit) for unit in unit_ids]
            metrics_df.loc[:, ['run_pval_rf', 'run_mod_rf']] = \
                    [self.get_running_modulation(unit, self.get_preferred_condition(unit)) for unit in unit_ids]

            self._metrics = metrics_df

        return self._metrics


    def _get_stim_table_stats(self):

        """ Extract azimuths and elevations from stimulus table."""

        self._pos_y = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_pos_y] != 'null'][self._col_pos_y].unique())
        self._pos_x = np.sort(self.stimulus_conditions.loc[self.stimulus_conditions[self._col_pos_x] != 'null'][self._col_pos_x].unique())


    def _get_rf(self, unit_id):

        """ Extract the receptive field for one unit

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        receptive_field - 9 x 9 numpy array

        """

        return self.receptive_fields['spike_count'].sel(unit_id=unit_id).data


    def _response_by_stimulus_position(self, dataset, presentations,
        row_key='Pos_y', column_key='Pos_x',
        unit_key='unit_id', time_key='time_relative_to_stimulus_onset',
        spike_count_key='spike_count'):

        """ Calculate the unit's response to different locations
        of the Gabor patch

        Params:
        -------
        dataset - xarray dataset of binned spike counts for each trial
        presentations - list of presentation_ids

        Returns:
        -------
        dataset - xarray dataset of receptive fields

        """

        dataset = dataset.copy()
        dataset['spike_counts'] = dataset['spike_counts'].sum(dim=time_key)
        dataset = dataset.drop(time_key)

        dataset[row_key] = presentations.loc[:, row_key]
        dataset[column_key] = presentations.loc[:, column_key]
        dataset = dataset.to_dataframe()

        dataset = dataset.reset_index(unit_key).groupby([row_key, column_key, unit_key]).sum()

        return dataset.rename(columns={'spike_counts': spike_count_key}).to_xarray()


    def _get_rf_stats(self, unit_id):

        """ Calculate a variety of metrics for one unit's receptive field

        Params:
        -------
        unit_id - unique ID for the unit of interest

        Returns:
        -------
        azimuth - preferred azimuth
        elevation - preferred elevation
        width - receptive field width 
        height - receptive field height
        area - receptive field area
        exists - True if significant receptive field is present
        on_screen - True if the receptive field is away from the screen edge

        """

        RF = self._get_rf(unit_id)

        """ Calculation goes here """

        azimuth = np.nan
        elevation = np.nan
        width = np.nan
        height = np.nan
        area = np.nan
        exists = np.nan
        on_screen = np.nan

        return azimuth, elevation, width, height, area, exists, on_screen