import numpy as np
import pandas as pd
from six import string_types
import scipy.ndimage as ndi
import scipy.stats as st
from scipy.optimize import curve_fit, leastsq

import matplotlib.pyplot as plt

from ...chisquare_categorical import chisq_from_stim_table

from .stimulus_analysis import StimulusAnalysis

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ReceptiveFieldMapping(StimulusAnalysis):
    """
    A class for computing single-unit metrics from the receptive field mapping stimulus of an ecephys session NWB file.

    To use, pass in a EcephysSession object::
        session = EcephysSession.from_nwb_path('/path/to/my.nwb')
        rf_analysis = ReceptiveFieldMapping(session)

    or, alternatively, pass in the file path::
        rf_analysis = ReceptiveFieldMapping('/path/to/my.nwb')

    You can also pass in a unit filter dictionary which will only select units with certain properties. For example
    to get only those units which are on probe C and found in the VISp area::
        rf_analysis = ReceptiveFieldMapping(session, filter={'location': 'probeC', 'structure_acronym': 'VISp'})

    To get a table of the individual unit metrics ranked by unit ID::
        metrics_table_df = rf_analysis.metrics()

    """

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
            self._stimulus_key = 'receptive_field_mapping'

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
    def number_elevations(self):
        """ Number of stimulus elevations """
        if self._pos_y is None:
            self._get_stim_table_stats()

        return len(self._pos_y)

    @property
    def number_azimuths(self):
        """ Number of stimulus azimuths """
        if self._pos_x is None:
            self._get_stim_table_stats()

        return len(self._pos_y)

    @property
    def null_condition(self):
        """ Stimulus condition ID for null stimulus (not used, so set to -1) """
        return -1

    @property
    def receptive_fields(self):
        """ Spatial receptive fields for N units (9 x 9 x N matrix of responses) """
        if self._rf_matrix is None:

            bin_edges = np.linspace(0, 0.249, 249)

            self.stim_table.loc[:, 'Pos_y'] = 40.0 - self.stim_table['Pos_y']

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
                ('p_value_rf', bool), 
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

            unit_ids = self.unit_ids
        
            metrics_df = self.empty_metrics_table()

            if len(self.stim_table) > 0:

                print('Calculating metrics for ' + self.name)

                metrics_df.loc[:, ['azimuth_rf',
                                   'elevation_rf',
                                   'width_rf',
                                   'height_rf',
                                   'area_rf',
                                   'p_value_rf',
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
        azimuth - preferred azimuth in degrees
            * based on center of mass of thresholded RF

        elevation - preferred elevation in degrees
            * based on center of mass of thresholded RF
        
        width - receptive field width in degrees
            * based on Gaussian fit
        
        height - receptive field height in degrees
            * based on Gaussian fit
        
        area - receptive field area in degrees^2
            * based on thresholded RF area
        
        p_value - probability that a significant receptive field is present
            * based on categorical chi-square test
        
        on_screen - True if the receptive field is away from the screen edge
            - based on Gaussian fit

        """

        rf = self._get_rf(unit_id)
        spikes_per_trial = self.presentationwise_statistics.xs(unit_id, level=1)['spike_counts'].values

        if np.sum(spikes_per_trial) < self._params['minimum_spike_count']:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        p_value = chisq_from_stim_table(self.stim_table,
                                       ['Pos_x','Pos_y'],
                                       np.expand_dims(spikes_per_trial,1))

        rf_thresh, azimuth, elevation, area = threshold_rf(rf, self._params['mask_threshold'])

        if is_rf_inverted(rf_thresh):
            rf = invert_rf(rf)

        (peak_height, center_y, center_x, width_y, width_x), success = fit_2d_gaussian(rf)
        on_screen = rf_on_screen(rf, center_y, center_x)

        height = width_y #* self._params['stimulus_step_size']
        width = width_x #* self._params['stimulus_step_size']

        return azimuth, elevation, width, height, area, p_value[0], on_screen


    ## VISUALIZATION ##

    def plot_raster(self, stimulus_condition_id, unit_id):
    
        """ Plot raster for one condition and one unit """

        idx_elev = np.where(self.elevations == self.stimulus_conditions.loc[stimulus_condition_id][self._col_pos_y])[0]
        idx_azi = np.where(self.azimuths == self.stimulus_conditions.loc[stimulus_condition_id][self._col_pos_x])[0]
        
        if len(idx_elev) == len(idx_azi) == 1:
     
            presentation_ids = \
                self.presentationwise_statistics.xs(unit_id, level=1)\
                [self.presentationwise_statistics.xs(unit_id, level=1)\
                ['stimulus_condition_id'] == stimulus_condition_id].index.values
            
            df = self.presentationwise_spike_times[ \
                (self.presentationwise_spike_times['stimulus_presentation_id'].isin(presentation_ids)) & \
                (self.presentationwise_spike_times['unit_id'] == unit_id) ]
                
            x = df.index.values - self.stim_table.loc[df.stimulus_presentation_id].start_time
            _, y = np.unique(df.stimulus_presentation_id, return_inverse=True) 
            
            idx_elev = self.number_elevations - idx_elev - 1 # reverse the elevation index so it matches the RF

            plt.subplot(self.number_elevations, self.number_azimuths, idx_elev*self.number_azimuths + idx_azi + 1)
            plt.scatter(x, y, c='k', s=1, alpha=0.25)
            plt.axis('off')


    def plot_rf(self, unit_id):

        """ Plot the spike counts across conditions """
        plt.imshow(self._get_rf(unit_id), cmap='Greys')
        plt.axis('off')


#### HELPER FUNCTIONS ####

def gaussian_function_2d(peak_height, center_y, center_x, width_y, width_x):
    
    """Returns a 2D Gaussian function
    
    Parameters:
    -----------
    peak_height : peak of distribution
    center_y : y-coordinate of distribution center
    center_x : x-coordinate of distribution center
    width_y : width of distribution along x-axis
    width_x : width of distribution along y-axis
    
    Returns:
    --------
    f(x,y) : function
        Returns the value of the distribution at a particular x,y coordinate
    
    """
    
    return lambda x,y: peak_height \
                       * np.exp( \
                       -( \
                         ((center_y - y) / width_y)**2 \
                       + ((center_x - x) / width_x)**2 \
                        ) \
                        / 2 \
                        )


def gaussian_moments_2d(data):
    
    """
    Finds the moments of a 2D Gaussian distribution, given an input matrix
    
    Parameters:
    -----------
    data - numpy.ndarray
        2D matrix
        
    Returns:
    --------
    peak_height : peak of distribution
    center_y : y-coordinate of distribution center
    center_x : x-coordinate of distribution center
    width_y : width of distribution along x-axis
    width_x : width of distribution along y-axis

    """
    
    total = data.sum()
    height = data.max()
    
    Y, X = np.indices(data.shape)
    center_y = (Y*data).sum()/total
    center_x = (X*data).sum()/total
    
    col = data[:, int(center_x)]    
    row = data[int(center_y), :]

    width_y = np.sqrt(np.abs((np.arange(row.size)-center_y)**2*row).sum()/row.sum())
    width_x = np.sqrt(np.abs((np.arange(col.size)-center_x)**2*col).sum()/col.sum())

    return height, center_y, center_x, width_y, width_x


def fit_2d_gaussian(matrix):
    
    """
    Fits a receptive field with a 2-dimensional Gaussian distribution
    
    Parameters:
    -----------
    matrix - numpy.ndarray
        2D matrix of spike counts
        
    Returns:
    --------
    parameters - tuple
        peak_height : peak of distribution
        center_y : y-coordinate of distribution center
        center_x : x-coordinate of distribution center
        width_y : width of distribution along x-axis
        width_x : width of distribution along y-axis
    success - bool
        True if a fit was found, False otherwise

    """
    

    params = gaussian_moments_2d(matrix)
    errorfunction = lambda p: np.ravel(gaussian_function_2d(*p)(*np.indices(matrix.shape)) - matrix)
    fit_params, ier = leastsq(errorfunction, params)
    success = True if ier < 5 else False
    
    return fit_params, success


def is_rf_inverted(rf_thresh):
    """
    Checks if the receptive field mapping timulus is suppressing or exciting the cell

    Parameters:
    -----------
    rf_thresh - matrix of spike counts at each stimulus position

    Returns:
    --------
    bool - True if the receptive field is inverted

    """

    edge_mask = np.zeros(rf_thresh.shape)

    edge_mask[:,0] = 1
    edge_mask[:,-1] = 1
    edge_mask[0,:] = 1
    edge_mask[-1,:] = 1

    num_edge_pixels = np.sum(rf_thresh * edge_mask)

    return num_edge_pixels > np.sum(edge_mask) / 2


def invert_rf(rf):

    """
    Creates an inverted version of the receptive field

    Parameters:
    ----------
    rf - matrix of spike counts at each stimulus position

    Returns:
    --------
    rf_inverted - new RF matrix

    """
    
    return np.max(rf) - rf



def threshold_rf(rf, threshold):
    
    """
    Creates a spatial mask based on the receptive field peak, and returns
    the x, y coordinates of the center of mass, as well as the area
    
    Parameters:
    -----------
    rf - numpy.ndarray
        2D matrix of spike counts
    threshold - float
        Threshold as ratio of the RF's standard deviation
        
    Returns:
    --------
    threshold_rf - numpy.ndarray
        Thresholded version of the original RF
    center_x - float
        x-coordinate of mask center of mass
    center_y - float
        y-coordinate of mask center of mass
    area - float
        area of mask
    
    """
    
    threshold_value = np.max(rf) - np.std(rf) * threshold
        
    rf_thresh = np.zeros(rf.shape, dtype='bool')
    rf_thresh[rf > threshold_value] = True
    
    labels, num_features = ndi.label(rf_thresh)
    
    best_label = np.argmax(ndi.maximum(rf, labels=labels, index=np.unique(labels)))

    labels[labels != best_label] = 0
    labels[labels > 0] = 1
    
    center_y, center_x = ndi.measurements.center_of_mass(labels)
    area = float(np.sum(labels))

    return labels, np.around(center_x,4), np.around(center_y,4), area


def rf_on_screen(rf, center_y, center_x):

    return 0 < center_y < rf.shape[0] and 0 < center_x < rf.shape[1]
