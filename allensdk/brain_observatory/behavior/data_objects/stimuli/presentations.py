import collections
from pathlib import Path
from typing import Optional, List, Dict, Union

import pandas as pd
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.core import DataObject
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.core import \
    NwbReadableInterface
from allensdk.brain_observatory.behavior.data_files.stimulus_file import \
    StimulusFileReadableInterface
from allensdk.core import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.stimulus_processing import \
    get_stimulus_presentations, get_stimulus_metadata, is_change_event
from allensdk.brain_observatory.nwb import \
    create_stimulus_presentation_time_interval


class Presentations(DataObject, StimulusFileReadableInterface,
                    NwbReadableInterface, NwbWritableInterface):
    """Stimulus presentations"""
    def __init__(self, presentations: pd.DataFrame,
                 columns_to_rename: Optional[Dict[str, str]] = None,
                 column_list: Optional[List[str]] = None,
                 sort_columns: bool = True):
        """

        Parameters
        ----------
        presentations: The stimulus presentations table
        columns_to_rename: Optional dict mapping
            old column name -> new column name
        column_list: Optional list of columns to include.
            This will reorder the columns.
        sort_columns: Whether to sort the columns by name
        """
        if columns_to_rename is not None:
            presentations = presentations.rename(columns=columns_to_rename)
        if column_list is not None:
            presentations = presentations[column_list]
        if sort_columns:
            presentations = presentations[sorted(presentations.columns)]
        presentations = presentations.reset_index(drop=True)
        presentations.index = pd.Index(
            range(presentations.shape[0]), name='stimulus_presentations_id',
            dtype='int')
        super().__init__(name='presentations', value=presentations)

    def to_nwb(self,
               nwbfile: NWBFile,
               stimulus_name_column='stimulus_name') -> NWBFile:
        """Adds a stimulus table (defining stimulus characteristics for each
        time point in a session) to an nwbfile as TimeIntervals.

        Parameters
        ----------
        nwbfile
        stimulus_name_column: The column in the dataframe denoting the
            stimulus name
        """
        stimulus_table = self.value.copy()

        ts = nwbfile.processing['stimulus'].get_data_interface('timestamps')
        stimulus_names = stimulus_table[stimulus_name_column].unique()

        for stim_name in sorted(stimulus_names):
            specific_stimulus_table = \
                stimulus_table[
                    stimulus_table[stimulus_name_column] == stim_name]
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
    def from_nwb(cls, nwbfile: NWBFile, add_is_change: bool = True,
                 column_list: Optional[List[str]] = None) -> "Presentations":
        """

        Parameters
        ----------
        nwbfile
        add_is_change: Whether to add a column denoting whether the current
            row represents a stimulus in which there was a change event
        column_list: The columns and order of columns
            in the final dataframe.

        Returns
        -------
        `Presentations` instance
        """
        columns_to_ignore = {'tags', 'timeseries', 'tags_index',
                             'timeseries_index'}

        presentation_dfs = []
        for interval_name, interval in nwbfile.intervals.items():
            if interval_name.endswith('_presentations'):
                presentations = collections.defaultdict(list)
                for col in interval.columns:
                    if col.name not in columns_to_ignore:
                        presentations[col.name].extend(col.data[:])
                df = pd.DataFrame(presentations).replace({'N/A': ''})
                presentation_dfs.append(df)

        table = pd.concat(presentation_dfs, sort=False)
        table = table.astype(
            {c: 'int64' for c in table.select_dtypes(include='int')})
        table = table.sort_values(by=["start_time"])

        table = table.reset_index(drop=True)
        table.index = table.index.astype('int64')

        if add_is_change:
            table['is_change'] = is_change_event(stimulus_presentations=table)
        return Presentations(presentations=table, column_list=column_list)

    @classmethod
    def from_stimulus_file(
            cls, stimulus_file: BehaviorStimulusFile,
            stimulus_timestamps: StimulusTimestamps,
            limit_to_images: Optional[List] = None,
            column_list: Optional[List[str]] = None,
            fill_omitted_values=True
    ) -> "Presentations":
        """Get stimulus presentation data.

        :param stimulus_file
        :param limit_to_images
            Only return images given by these image names
        :param stimulus_timestamps
        :param column_list: The columns and order of columns
            in the final dataframe
        :param fill_omitted_values: Whether to fill stop_time and duration
            for omitted frames


        :returns: pd.DataFrame --
            Table whose rows are stimulus presentations
            (i.e. a given image, for a given duration, typically 250 ms)
            and whose columns are presentation characteristics.
        """
        stimulus_timestamps = stimulus_timestamps.value
        data = stimulus_file.data
        raw_stim_pres_df = get_stimulus_presentations(
            data, stimulus_timestamps)
        raw_stim_pres_df = raw_stim_pres_df.drop(columns=['index'])

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
                raise ValueError("All values for 'orientation' and "
                                 "'image_name are null.")

        stimulus_metadata_df = get_stimulus_metadata(data)

        idx_name = raw_stim_pres_df.index.name
        stimulus_index_df = (
            raw_stim_pres_df
            .reset_index()
            .merge(stimulus_metadata_df.reset_index(),
                   on=["image_name"])
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
        stim_pres_df = cls._postprocess(
            presentations=stim_pres_df,
            fill_omitted_values=fill_omitted_values)
        return Presentations(presentations=stim_pres_df,
                             column_list=column_list)

    @classmethod
    def from_path(cls,
                  path: Union[str, Path],
                  exclude_columns: Optional[List[str]] = None,
                  columns_to_rename: Optional[Dict[str, str]] = None,
                  sort_columns: bool = True
                  ) -> "Presentations":
        """
        Reads the table directly from a precomputed csv

        Parameters
        -----------
        path: Path to load table from
        exclude_columns: Columns to exclude
        columns_to_rename: Optional d ict mapping
            old column name -> new column name
        sort_columns: Whether to sort the columns by name
        Returns
        -------
        Presentations instance
        """
        path = Path(path)
        df = pd.read_csv(path)
        exclude_columns = exclude_columns if exclude_columns else []
        df = df[[c for c in df if c not in exclude_columns]]
        return Presentations(presentations=df,
                             columns_to_rename=columns_to_rename,
                             sort_columns=sort_columns)

    @classmethod
    def _postprocess(cls, presentations: pd.DataFrame,
                     fill_omitted_values=True,
                     omitted_time_duration: float = 0.25) \
            -> pd.DataFrame:
        """
        Optionally fill missing values for omitted flashes (no need when
            reading from NWB since already filled)

        Parameters
        ----------
        presentations
            Presentations df
        fill_omitted_values
            Whether to fill stop time and duration for omitted flashes
        omitted_time_duration
            Amount of time a stimuli is omitted for in seconds"""
        df = presentations
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
