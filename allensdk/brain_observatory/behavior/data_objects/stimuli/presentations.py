from typing import Optional, List

import pandas as pd
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    StimulusTimestamps
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    StimulusFileReadableInterface, NwbReadableInterface
from allensdk.brain_observatory.behavior.data_objects.base \
    .writable_interfaces import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.stimulus_processing import \
    get_stimulus_presentations, get_stimulus_metadata, is_change_event
from allensdk.brain_observatory.nwb import \
    create_stimulus_presentation_time_interval, get_column_name
from allensdk.brain_observatory.nwb.nwb_api import NwbApi


class Presentations(DataObject, StimulusFileReadableInterface,
                    NwbReadableInterface, NwbWritableInterface):
    """Stimulus presentations"""
    def __init__(self, presentations: pd.DataFrame):
        super().__init__(name='presentations', value=presentations)

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        """Adds a stimulus table (defining stimulus characteristics for each
        time point in a session) to an nwbfile as TimeIntervals.
        """
        stimulus_table = self.value.copy()

        ts = nwbfile.processing['stimulus'].get_data_interface('timestamps')
        possible_names = {'stimulus_name', 'image_name'}
        stimulus_name_column = get_column_name(stimulus_table.columns,
                                               possible_names)
        stimulus_names = stimulus_table[stimulus_name_column].unique()

        for stim_name in sorted(stimulus_names):
            specific_stimulus_table = stimulus_table[stimulus_table[
                                                         stimulus_name_column] == stim_name]  # noqa: E501
            # Drop columns where all values in column are NaN
            cleaned_table = specific_stimulus_table.dropna(axis=1, how='all')
            # For columns with mixed strings and NaNs, fill NaNs with 'N/A'
            for colname, series in cleaned_table.items():
                types = set(series.map(type))
                if len(types) > 1 and str in types:
                    series.fillna('N/A', inplace=True)
                    cleaned_table[colname] = series.transform(str)

            interval_description = (f"Presentation times and stimuli details "
                                    f"for '{stim_name}' stimuli. "
                                    f"\n"
                                    f"Note: image_name references "
                                    f"control_description in "
                                    f"stimulus/templates")
            presentation_interval = create_stimulus_presentation_time_interval(
                name=f"{stim_name}_presentations",
                description=interval_description,
                columns_to_add=cleaned_table.columns
            )

            for row in cleaned_table.itertuples(index=False):
                row = row._asdict()

                presentation_interval.add_interval(
                    **row, tags='stimulus_time_interval', timeseries=ts)

            nwbfile.add_time_intervals(presentation_interval)

        return nwbfile

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "Presentations":
        # Note: using NwbApi class because ecephys uses this method
        # TODO figure out how behavior and ecephys can share this method
        nwbapi = NwbApi.from_nwbfile(nwbfile=nwbfile)
        df = nwbapi.get_stimulus_presentations()

        df['is_change'] = is_change_event(stimulus_presentations=df)
        df = cls._postprocess(presentations=df, fill_omitted_values=False)
        return Presentations(presentations=df)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: StimulusFile,
            stimulus_timestamps: StimulusTimestamps,
            limit_to_images: Optional[List] = None) -> "Presentations":
        """Get stimulus presentation data.

        :param stimulus_file
        :param limit_to_images
            Only return images given by these image names
        :param stimulus_timestamps


        :returns: pd.DataFrame --
            Table whose rows are stimulus presentations
            (i.e. a given image, for a given duration, typically 250 ms)
            and whose columns are presentation characteristics.
        """
        stimulus_timestamps = stimulus_timestamps.value
        data = stimulus_file.data
        raw_stim_pres_df = get_stimulus_presentations(
            data, stimulus_timestamps)

        # Fill in nulls for image_name
        # This makes two assumptions:
        #   1. Nulls in `image_name` should be "gratings_<orientation>"
        #   2. Gratings are only present (or need to be fixed) when all
        #      values for `image_name` are null.
        if pd.isnull(raw_stim_pres_df["image_name"]).all():
            if ~pd.isnull(raw_stim_pres_df["orientation"]).all():
                raw_stim_pres_df["image_name"] = (
                    raw_stim_pres_df["orientation"]
                    .apply(lambda x: f"gratings_{x}"))
            else:
                raise ValueError("All values for 'orentation' and 'image_name'"
                                 " are null.")

        stimulus_metadata_df = get_stimulus_metadata(data)

        idx_name = raw_stim_pres_df.index.name
        stimulus_index_df = (
            raw_stim_pres_df
            .reset_index()
            .merge(stimulus_metadata_df.reset_index(), on=["image_name"])
            .set_index(idx_name))
        stimulus_index_df = (
            stimulus_index_df[["image_set", "image_index", "start_time",
                               "phase", "spatial_frequency"]]
            .rename(columns={"start_time": "timestamps"})
            .sort_index()
            .set_index("timestamps", drop=True))
        stim_pres_df = raw_stim_pres_df.merge(
            stimulus_index_df, left_on="start_time", right_index=True,
            how="left")
        if len(raw_stim_pres_df) != len(stim_pres_df):
            raise ValueError("Length of `stim_pres_df` should not change after"
                             f" merge; was {len(raw_stim_pres_df)}, now "
                             f" {len(stim_pres_df)}.")

        stim_pres_df['is_change'] = is_change_event(
            stimulus_presentations=stim_pres_df)

        # Sort columns then drop columns which contain only all NaN values
        stim_pres_df = \
            stim_pres_df[sorted(stim_pres_df)].dropna(axis=1, how='all')
        if limit_to_images is not None:
            stim_pres_df = \
                stim_pres_df[stim_pres_df['image_name'].isin(limit_to_images)]
            stim_pres_df.index = pd.Int64Index(
                range(stim_pres_df.shape[0]), name=stim_pres_df.index.name)
        stim_pres_df = cls._postprocess(presentations=stim_pres_df)
        return Presentations(presentations=stim_pres_df)

    @classmethod
    def _postprocess(cls, presentations: pd.DataFrame,
                     fill_omitted_values=True,
                     omitted_time_duration: float = 0.25) \
            -> pd.DataFrame:
        """
        1. Filter/rearrange columns
        2. Optionally fill missing values for omitted flashes (no need when
            reading from NWB since already filled)

        Parameters
        ----------
        presentations
            Presentations df
        fill_omitted_values
            Whether to fill stop time and duration for omitted flashes
        omitted_time_duration
            Amount of time a stimuli is omitted for in seconds"""

        def _filter_arrange_columns(df: pd.DataFrame):
            df = df.drop(['index'], axis=1, errors='ignore')
            df = df[['start_time', 'stop_time',
                     'duration',
                     'image_name', 'image_index',
                     'is_change', 'omitted',
                     'start_frame', 'end_frame',
                     'image_set']]
            return df

        df = _filter_arrange_columns(df=presentations)
        if fill_omitted_values:
            cls._fill_missing_values_for_omitted_flashes(
                df=df, omitted_time_duration=omitted_time_duration)
        return df

    @staticmethod
    def _fill_missing_values_for_omitted_flashes(
            df: pd.DataFrame, omitted_time_duration: float = 0.25) \
            -> pd.DataFrame:
        """
        This function sets the stop time for a row that is an omitted
        stimulus. An omitted stimulus is a stimulus where a mouse is
        shown only a grey screen and these last for 250 milliseconds.
        These do not include a stop_time or end_frame like other stimuli in
        the stimulus table due to design choices.

        Parameters
        ----------
        df
            Stimuli presentations dataframe
        omitted_time_duration
            Amount of time a stimulus is omitted for in seconds
        """
        omitted = df['omitted']
        df.loc[omitted, 'stop_time'] = \
            df.loc[omitted, 'start_time'] + omitted_time_duration
        df.loc[omitted, 'duration'] = omitted_time_duration
        return df
