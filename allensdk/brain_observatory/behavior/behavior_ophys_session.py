import numpy as np
import pandas as pd
import xarray as xr
from typing import Any, Optional
import logging


from allensdk.brain_observatory.session_api_utils import ParamsMixin
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorOphysNwbApi, BehaviorOphysLimsApi)
from allensdk.deprecated import legacy
from allensdk.brain_observatory.behavior.trials_processing import (
    calculate_reward_rate)
from allensdk.brain_observatory.behavior.dprime import (
    get_rolling_dprime, get_trial_count_corrected_false_alarm_rate,
    get_trial_count_corrected_hit_rate)
from allensdk.brain_observatory.behavior.dprime import (
    get_hit_rate, get_false_alarm_rate)
from allensdk.brain_observatory.behavior.image_api import Image, ImageApi


class BehaviorOphysSession(ParamsMixin):
    """Represents data from a single Visual Behavior Ophys imaging session.
    Can be initialized with an api that fetches data, or by using class methods
    `from_lims` and `from_nwb_path`.
    """

    @classmethod
    def from_lims(cls, ophys_experiment_id: int,
                  eye_tracking_z_threshold: float = 3.0,
                  eye_tracking_dilation_frames: int = 2
                  ) -> "BehaviorOphysSession":
        return cls(api=BehaviorOphysLimsApi(ophys_experiment_id),
                   eye_tracking_z_threshold=eye_tracking_z_threshold,
                   eye_tracking_dilation_frames=eye_tracking_dilation_frames)

    @classmethod
    def from_nwb_path(
            cls, nwb_path: str, **api_kwargs: Any) -> "BehaviorOphysSession":
        api_kwargs["filter_invalid_rois"] = api_kwargs.get(
            "filter_invalid_rois", True)
        return cls(api=BehaviorOphysNwbApi.from_path(
            path=nwb_path, **api_kwargs))

    def __init__(self, api=None,
                 eye_tracking_z_threshold: float = 3.0,
                 eye_tracking_dilation_frames: int = 2):
        """
        Parameters
        ----------
        api : object, optional
            The backend api used by the session object to get behavior ophys
            data, by default None.
        eye_tracking_z_threshold : float, optional
            Determines the z-score threshold used for processing
            `eye_tracking` data, by default 3.0.
        eye_tracking_dilation_frames : int, optional
            Determines the number of adjacent frames that will be marked
            as 'likely_blink' when performing blink detection for
            `eye_tracking` data, by default 2
        """
        super().__init__(ignore={'api'})

        self.api = api
        # Initialize attributes to be lazily evaluated
        self._stimulus_timestamps = None
        self._ophys_timestamps = None
        self._metadata = None
        self._dff_traces = None
        self._cell_specimen_table = None
        self._running_speed = None
        self._running_data_df = None
        self._raw_running_speed = None
        self._raw_running_data_df = None
        self._stimulus_presentations = None
        self._stimulus_templates = None
        self._licks = None
        self._rewards = None
        self._task_parameters = None
        self._trials = None
        self._corrected_fluorescence_traces = None
        self._motion_correction = None
        self._segmentation_mask_image = None
        self._eye_tracking = None

        # eye_tracking params
        self._eye_tracking_z_threshold = eye_tracking_z_threshold
        self._eye_tracking_dilation_frames = eye_tracking_dilation_frames

    # Using properties rather than initializing attributes to take advantage
    # of API-level cache and not introduce a lot of overhead when the
    # class is initialized (sometimes these calls can take a while)
    @property
    def ophys_experiment_id(self) -> int:
        """Unique identifier for this experimental session.
        :rtype: int
        """
        return self.api.get_ophys_experiment_id()

    @property
    def behavior_session_id(self) -> Optional[int]:
        """Returns the unique identifier for the behavior session
        associated with this experiment, if applicable.
        """
        return self.api.get_behavior_session_id()

    @property
    def max_projection(self) -> Image:
        """2D max projection image.
        :rtype: allensdk.brain_observatory.behavior.image_api.Image
        """
        return self.get_max_projection()

    @property
    def stimulus_timestamps(self) -> np.ndarray:
        """Timestamps associated with stimulus presentations on the
        monitor (corrected for monitor delay).
        :rtype: numpy.ndarray
        """
        if self._stimulus_timestamps is None:
            self._stimulus_timestamps = self.api.get_stimulus_timestamps()
        return self._stimulus_timestamps

    @stimulus_timestamps.setter
    def stimulus_timestamps(self, value):
        self._stimulus_timestamps = value

    @property
    def ophys_timestamps(self) -> np.ndarray:
        """Timestamps associated with frames captured by the microscope
        :rtype: numpy.ndarray
        """
        if self._ophys_timestamps is None:
            self._ophys_timestamps = self.api.get_ophys_timestamps()
        return self._ophys_timestamps

    @ophys_timestamps.setter
    def ophys_timestamps(self, value):
        self._ophys_timestamps = value

    @property
    def metadata(self) -> dict:
        """Dictionary of session-specific metadata.
        :rtype: dict
        """
        if self._metadata is None:
            self._metadata = self.api.get_metadata()
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def dff_traces(self) -> pd.DataFrame:
        """Traces of dff organized into a dataframe; index is the cell roi ids.
        :rtype: pandas.DataFrame
        """
        if self._dff_traces is None:
            self._dff_traces = self.api.get_dff_traces()
        return self._dff_traces

    @dff_traces.setter
    def dff_traces(self, value):
        self._dff_traces = value

    @property
    def cell_specimen_table(self) -> pd.DataFrame:
        """Cell roi information organized into a dataframe; index is the cell
        roi ids.
        :rtype: pandas.DataFrame
        """
        if self._cell_specimen_table is None:
            self._cell_specimen_table = self.api.get_cell_specimen_table()
        return self._cell_specimen_table

    @cell_specimen_table.setter
    def cell_specimen_table(self, value):
        self._cell_specimen_table = value

    @property
    def running_speed(self) -> pd.DataFrame:
        """Running speed of mouse, filtered with 10Hz low pass filter,

        Returns
        -------
        self._running_speed: pd.DataFrame
            colums:
            - timestamps [seconds]
            - values [cm/s]

        Notes
        -----
        To get the unfiltered running speed, use `raw_running_speed`.
        """
        if self._running_speed is None:
            rs_tuple = self.api.get_running_speed()
            self._running_speed = pd.DataFrame(
                    {"timestamps": rs_tuple.timestamps,
                     "values": rs_tuple.values})
        return self._running_speed

    @running_speed.setter
    def running_speed(self, value):
        self._running_speed = value

    @property
    def raw_running_speed(self) -> pd.DataFrame:
        """Running speed of mouse, unfiltered

        Returns
        -------
        self._running_speed: pd.DataFrame
            colums:
            - timestamps [seconds]
            - values [cm/s]

        Notes
        -----
        To get the filtered running speed, use `running_speed`.
        """
        if self._raw_running_speed is None:
            rs_tuple = self.api.get_running_speed(lowpass=False)
            self._raw_running_speed = pd.DataFrame(
                    {"timestamps": rs_tuple.timestamps,
                     "values": rs_tuple.values})
        return self._raw_running_speed

    @raw_running_speed.setter
    def raw_running_speed(self, value):
        self._raw_running_speed = value

    @property
    def stimulus_presentations(self) -> pd.DataFrame:
        """Table whose rows are stimulus presentations (i.e. a given image,
        for a given duration, typically 250 ms) and whose columns are
        presentation characteristics.
        :rtype: pandas.DataFrame
        """
        if self._stimulus_presentations is None:
            self._stimulus_presentations = (
                self.api.get_stimulus_presentations())
        return self._stimulus_presentations

    @stimulus_presentations.setter
    def stimulus_presentations(self, value):
        self._stimulus_presentations = value

    @property
    def stimulus_templates(self) -> dict:
        """A dictionary containing the stimulus images presented during the
        session keys are data set names, and values are 3D numpy arrays.
        :rtype: dict
        """
        if self._stimulus_templates is None:
            self._stimulus_templates = self.api.get_stimulus_templates()
        return self._stimulus_templates

    @stimulus_templates.setter
    def stimulus_templates(self, value):
        self._stimulus_templates = value

    @property
    def licks(self) -> pd.DataFrame:
        """A dataframe containing lick timestamps.
        :rtype: pandas.DataFrame
        """
        if self._licks is None:
            self._licks = self.api.get_licks()
        return self._licks

    @licks.setter
    def licks(self, value):
        self._licks = value

    @property
    def rewards(self) -> pd.DataFrame:
        """A dataframe containing timestamps of delivered rewards.
        :rtype: pandas.DataFrame
        """
        if self._rewards is None:
            self._rewards = self.api.get_rewards()
        return self._rewards

    @rewards.setter
    def rewards(self, value):
        self._rewards = value

    @property
    def task_parameters(self) -> dict:
        """A dictionary containing parameters used to define the task runtime
        behavior.
        :rtype: dict
        """
        if self._task_parameters is None:
            self._task_parameters = self.api.get_task_parameters()
        return self._task_parameters

    @task_parameters.setter
    def task_parameters(self, value):
        self._task_parameters = value

    @property
    def trials(self) -> pd.DataFrame:
        """A dataframe containing behavioral trial start/stop times, and trial
        data
        :rtype: pandas.DataFrame"""
        if self._trials is None:
            self._trials = self.api.get_trials()
        return self._trials

    @trials.setter
    def trials(self, value):
        self._trials = value

    @property
    def corrected_fluorescence_traces(self) -> pd.DataFrame:
        """The motion-corrected fluorescence traces organized into a dataframe;
        index is the cell roi ids.
        :rtype: pandas.DataFrame
        """
        if self._corrected_fluorescence_traces is None:
            self._corrected_fluorescence_traces = (
                self.api.get_corrected_fluorescence_traces())
        return self._corrected_fluorescence_traces

    @corrected_fluorescence_traces.setter
    def corrected_fluorescence_traces(self, value):
        self._corrected_fluorescence_traces = value

    @property
    def average_projection(self) -> pd.DataFrame:
        """2D image of the microscope field of view, averaged across the
        experiment
        :rtype: pandas.DataFrame
        """
        return self.get_average_projection()

    @property
    def motion_correction(self) -> pd.DataFrame:
        """A dataframe containing trace data used during motion correction
        computation
        :rtype: pandas.DataFrame
        """
        if self._motion_correction is None:
            self._motion_correction = self.api.get_motion_correction()
        return self._motion_correction

    @motion_correction.setter
    def motion_correction(self, value):
        self._motion_correction = value

    @property
    def segmentation_mask_image(self) -> Image:
        """An image with pixel value 1 if that pixel was included in an ROI,
        and 0 otherwise
        :rtype: allensdk.brain_observatory.behavior.image_api.Image
        """
        if self._segmentation_mask_image is None:
            self._segmentation_mask_image = self.get_segmentation_mask_image()
        return self._segmentation_mask_image

    @segmentation_mask_image.setter
    def segmentation_mask_image(self, value):
        self._segmentation_mask_image = value

    @property
    def eye_tracking(self) -> pd.DataFrame:
        """A dataframe containing ellipse fit parameters for the eye, pupil
        and corneal reflection (cr). Fits are derived from tracking points
        from a DeepLabCut model applied to video frames of a subject's
        right eye. Raw tracking points and raw video frames are not exposed
        by the SDK.

        Notes:
        - All columns starting with 'pupil_' represent ellipse fit parameters
          relating to the pupil.
        - All columns starting with 'eye_' represent ellipse fit parameters
          relating to the eyelid.
        - All columns starting with 'cr_' represent ellipse fit parameters
          relating to the corneal reflection, which is caused by an infrared
          LED positioned near the eye tracking camera.
        - All positions are in units of pixels.
        - All areas are in units of pixels^2
        - All values are in the coordinate space of the eye tracking camera,
          NOT the coordinate space of the stimulus display (i.e. this is not
          gaze location), with (0, 0) being the upper-left corner of the
          eye-tracking image.
        - The 'likely_blink' column is True for any row (frame) where the pupil
          fit failed OR eye fit failed OR an outlier fit was identified.
        - All ellipse fits are derived from tracking points that were output by
          a DeepLabCut model that was trained on hand-annotated data frome a
          subset of imaging sessions on optical physiology rigs.
        - Raw DeepLabCut tracking points are not publicly available.

        :rtype: pandas.DataFrame
        """
        params = {'eye_tracking_dilation_frames', 'eye_tracking_z_threshold'}

        if (self._eye_tracking is None) or self.needs_data_refresh(params):
            self._eye_tracking = self.api.get_eye_tracking(
                    z_threshold=self._eye_tracking_z_threshold,
                    dilation_frames=self._eye_tracking_dilation_frames)
            self.clear_updated_params(params)

        return self._eye_tracking

    @eye_tracking.setter
    def eye_tracking(self, value):
        self._eye_tracking = value

    @property
    def eye_tracking_rig_geometry(self) -> dict:
        """Get the eye tracking rig geometry
        associated with an ophys experiment"""
        return self.api.get_eye_tracking_rig_geometry()

    @property
    def eye_gaze_mapping_file_path(self) -> str:
        """Get h5 filepath containing eye gaze behavior of
        the experiment's subject"""
        return self.api.get_eye_gaze_mapping_file_path()

    def cache_clear(self) -> None:
        """Convenience method to clear the api cache, if applicable."""
        try:
            self.api.cache_clear()
        except AttributeError:
            logging.getLogger("BehaviorOphysSession").warning(
                "Attempted to clear API cache, but method `cache_clear`"
                f" does not exist on {self.api.__class__.__name__}")

    @property
    def roi_masks(self) -> pd.DataFrame:
        return self.cell_specimen_table[['cell_roi_id', 'roi_mask']]

    @legacy('Consider using "dff_traces" instead.')
    def get_dff_traces(self, cell_specimen_ids=None):

        if cell_specimen_ids is None:
            cell_specimen_ids = self.get_cell_specimen_ids()

        csid_table = \
            self.cell_specimen_table.reset_index()[['cell_specimen_id']]
        csid_subtable = csid_table[csid_table['cell_specimen_id'].isin(
            cell_specimen_ids)].set_index('cell_specimen_id')
        dff_table = csid_subtable.join(self.dff_traces, how='left')
        dff_traces = np.vstack(dff_table['dff'].values)
        timestamps = self.ophys_timestamps

        assert (len(cell_specimen_ids), len(timestamps)) == dff_traces.shape
        return timestamps, dff_traces

    @legacy()
    def get_cell_specimen_indices(self, cell_specimen_ids):
        return [self.cell_specimen_table.index.get_loc(csid)
                for csid in cell_specimen_ids]

    @legacy("Consider using cell_specimen_table['cell_specimen_id'] instead.")
    def get_cell_specimen_ids(self):
        cell_specimen_ids = self.cell_specimen_table.index.values

        if np.isnan(cell_specimen_ids.astype(float)).sum() == \
                len(self.cell_specimen_table):
            raise ValueError("cell_specimen_id values not assigned "
                             f"for {self.ophys_experiment_id}")
        return cell_specimen_ids

    def deserialize_image(self, sitk_image):
        '''
        Convert SimpleITK image returned by the api to an Image class:

        Args:
            sitk_image (SimpleITK image): image object returned by the api

        Returns
            img (allensdk.brain_observatory.behavior.image_api.Image)
        '''
        img = ImageApi.deserialize(sitk_image)
        return img

    def get_max_projection(self):
        """ Returns an image whose values are the maximum obtained
        values at each pixel of the ophys movie over time.

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            array-like interface to max projection image data and metadata
        """
        return self.deserialize_image(self.api.get_max_projection())

    def get_average_projection(self):
        """ Returns an image whose values are the average obtained
        values at each pixel of the ophys movie over time.

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            array-like interface to max projection image data and metadata
        """
        return self.deserialize_image(self.api.get_average_projection())

    def get_segmentation_mask_image(self):
        """ Returns an image with value 1 if the pixel was included
        in an ROI, and 0 otherwise

        Returns
        ----------
        allensdk.brain_observatory.behavior.image_api.Image:
            array-like interface to segmentation_mask image data and metadata

        """
        masks = self.api.get_roi_masks_by_cell_roi_id()
        mask_image_data = masks.any(dim='cell_roi_id').astype(int)
        mask_image = Image(
            data=mask_image_data.values,
            spacing=masks.attrs['spacing'],
            unit=masks.attrs['unit']
        )
        return mask_image

    def get_reward_rate(self):
        response_latency_list = []
        for _, t in self.trials.iterrows():
            valid_response_licks = \
                    [l for l in t.lick_times
                     if l - t.change_time >
                        self.task_parameters['response_window_sec'][0]]
            response_latency = (
                    float('inf')
                    if len(valid_response_licks) == 0
                    else valid_response_licks[0] - t.change_time)
            response_latency_list.append(response_latency)
        reward_rate = calculate_reward_rate(
                response_latency=response_latency_list,
                starttime=self.trials.start_time.values)
        reward_rate[np.isinf(reward_rate)] = float('nan')
        return reward_rate

    def get_rolling_performance_df(self):
        # Indices to build trial metrics dataframe:
        trials_index = self.trials.index
        not_aborted_index = \
            self.trials[np.logical_not(self.trials.aborted)].index

        # Initialize dataframe:
        performance_metrics_df = pd.DataFrame(index=trials_index)

        # Reward rate:
        performance_metrics_df['reward_rate'] = \
            pd.Series(self.get_reward_rate(), index=self.trials.index)

        # Hit rate raw:
        hit_rate_raw = get_hit_rate(
            hit=self.trials.hit,
            miss=self.trials.miss,
            aborted=self.trials.aborted)
        performance_metrics_df['hit_rate_raw'] = \
            pd.Series(hit_rate_raw, index=not_aborted_index)

        # Hit rate with trial count correction:
        hit_rate = get_trial_count_corrected_hit_rate(
                hit=self.trials.hit,
                miss=self.trials.miss,
                aborted=self.trials.aborted)
        performance_metrics_df['hit_rate'] = \
            pd.Series(hit_rate, index=not_aborted_index)

        # False-alarm rate raw:
        false_alarm_rate_raw = \
            get_false_alarm_rate(
                    false_alarm=self.trials.false_alarm,
                    correct_reject=self.trials.correct_reject,
                    aborted=self.trials.aborted)
        performance_metrics_df['false_alarm_rate_raw'] = \
            pd.Series(false_alarm_rate_raw, index=not_aborted_index)

        # False-alarm rate with trial count correction:
        false_alarm_rate = \
            get_trial_count_corrected_false_alarm_rate(
                    false_alarm=self.trials.false_alarm,
                    correct_reject=self.trials.correct_reject,
                    aborted=self.trials.aborted)
        performance_metrics_df['false_alarm_rate'] = \
            pd.Series(false_alarm_rate, index=not_aborted_index)

        # Rolling-dprime:
        rolling_dprime = get_rolling_dprime(hit_rate, false_alarm_rate)
        performance_metrics_df['rolling_dprime'] = \
            pd.Series(rolling_dprime, index=not_aborted_index)

        return performance_metrics_df

    def get_performance_metrics(self, engaged_trial_reward_rate_threshold=2):
        performance_metrics = {}
        performance_metrics['trial_count'] = len(self.trials)
        performance_metrics['go_trial_count'] = self.trials.go.sum()
        performance_metrics['catch_trial_count'] = self.trials.catch.sum()
        performance_metrics['hit_trial_count'] = self.trials.hit.sum()
        performance_metrics['miss_trial_count'] = self.trials.miss.sum()
        performance_metrics['false_alarm_trial_count'] = \
            self.trials.false_alarm.sum()
        performance_metrics['correct_reject_trial_count'] = \
            self.trials.correct_reject.sum()
        performance_metrics['auto_rewarded_trial_count'] = \
            self.trials.auto_rewarded.sum()
        performance_metrics['rewarded_trial_count'] = \
            self.trials.reward_time.apply(lambda x: not np.isnan(x)).sum()
        performance_metrics['total_reward_count'] = len(self.rewards)
        performance_metrics['total_reward_volume'] = self.rewards.volume.sum()

        rpdf = self.get_rpdf()
        engaged_trial_mask = (
                rpdf['reward_rate'] >
                engaged_trial_reward_rate_threshold)
        performance_metrics['maximum_reward_rate'] = \
            np.nanmax(rpdf['reward_rate'].values)
        performance_metrics['engaged_trial_count'] = (engaged_trial_mask).sum()
        performance_metrics['mean_hit_rate'] = \
            rpdf['hit_rate'].mean()
        performance_metrics['mean_hit_rate_uncorrected'] = \
            rpdf['hit_rate_raw'].mean()
        performance_metrics['mean_hit_rate_engaged'] = \
            rpdf['hit_rate'][engaged_trial_mask].mean()
        performance_metrics['mean_false_alarm_rate'] = \
            rpdf['false_alarm_rate'].mean()
        performance_metrics['mean_false_alarm_rate_uncorrected'] = \
            rpdf['false_alarm_rate_raw'].mean()
        performance_metrics['mean_false_alarm_rate_engaged'] = \
            rpdf['false_alarm_rate'][engaged_trial_mask].mean()
        performance_metrics['mean_dprime'] = \
            rpdf['rolling_dprime'].mean()
        performance_metrics['mean_dprime_engaged'] = \
            rpdf['rolling_dprime'][engaged_trial_mask].mean()
        performance_metrics['max_dprime'] = \
            rpdf['rolling_dprime'].max()
        performance_metrics['max_dprime_engaged'] = \
            rpdf['rolling_dprime'][engaged_trial_mask].max()

        return performance_metrics


if __name__ == "__main__":

    ophys_experiment_id = 789359614
    session = BehaviorOphysSession.from_lims(ophys_experiment_id)
    print(session.trials['reward_time'])
