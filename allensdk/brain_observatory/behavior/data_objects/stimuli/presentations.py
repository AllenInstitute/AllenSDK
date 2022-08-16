import collections
from pathlib import Path
from typing import Optional, List, Dict, Union

import numpy as np
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
        data = stimulus_file.data
        raw_stim_pres_df = get_stimulus_presentations(
            data, stimulus_timestamps.value)
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

        stim_pres_df['stimulus_block'] = 0
        stim_pres_df['stimulus_name'] = 'behavior'

        has_fingerprint_stimulus = \
            'fingerprint' in stimulus_file.data['items']['behavior']['items']
        if has_fingerprint_stimulus:
            stim_pres_df = cls._add_fingerprint_stimulus(
                stimulus_presentations=stim_pres_df,
                stimulus_file=stimulus_file,
                stimulus_timestamps=stimulus_timestamps
            )
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

    @staticmethod
    def _get_fingerprint_stimulus(
            stimulus_presentations: pd.DataFrame,
            stimulus_file: BehaviorStimulusFile,
            stimulus_timestamps: StimulusTimestamps
    ) -> pd.DataFrame:
        """The fingerprint stimulus is a movie used to trigger many neurons
        and is used to improve cell matching. This method adds rows for this
        stimulus to the stimulus presentations table

        Parameters
        ----------
        stimulus_presentations: stimulus presentations dataframe

        Returns
        ----------
        stimulus_presentations: stimulus presentations table for
        fingerprint stimulus
        """
        fingerprint_stim = (
            stimulus_file.data['items']['behavior']['items']['fingerprint']
            ['static_stimulus'])

        # start time is relative to session start. stop_time here
        # is assumed to be the last stop time of the last stimulus prior to the
        # fingerprint stimulus
        start_time = stimulus_presentations['stop_time'].max() + \
            fingerprint_stim['start_time']

        n_repeats = fingerprint_stim['runs']

        duration = fingerprint_stim['frame_length']
        movie_frame_rate = 1 / duration
        monitor_frame_rate = \
            stimulus_file.data['items']['behavior']['stim_config']['fps']

        # spontaneous + fingerprint indices relative to start of session
        frame_indices = np.array(
            stimulus_file.data['items']['behavior']['items']
            ['fingerprint']['frame_indices'])

        movie_length = int(len(fingerprint_stim['sweep_frames']) / n_repeats)

        # Start index within the spontaneous + fingerprint block
        movie_start_index = int(
            fingerprint_stim['start_time'] * monitor_frame_rate)

        res = []
        for repeat in range(n_repeats):
            for frame in range(movie_length):
                # 0-indexed frame indices relative to start of fingerprint
                # movie
                stimulus_frame_indices = \
                    np.array(fingerprint_stim['sweep_frames']
                             [frame + (repeat * movie_length)])
                start_frame, end_frame = frame_indices[
                    stimulus_frame_indices + movie_start_index]
                start_time, stop_time = \
                    stimulus_timestamps.value[[start_frame, end_frame + 1]]
                res.append({
                    'movie_frame_index': frame,
                    'start_time': start_time,
                    'stop_time': stop_time,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'repeat': repeat,
                    'duration': stop_time - start_time
                })
        res = pd.DataFrame(res)

        res['stimulus_block'] = \
            stimulus_presentations['stimulus_block'].max() \
            + 2     # + 2 since there is a gap before this stimulus
        res['stimulus_name'] = 'natural_movie_one'
        return res

    @classmethod
    def _get_spontaneous_stimulus(
            cls,
            stimulus_presentations_table: pd.DataFrame
    ) -> pd.DataFrame:
        """The spontaneous stimulus is a gray screen shown in between
        different stimulus blocks. This method finds any gaps in the stimulus
        presentations. These gaps are assumed to be spontaneous stimulus.

        Raises
        ------
        RuntimeError if there are any gaps in stimulus blocks > 1

        Returns
        -------
        pd.DataFrame: a dataframe with a single row for each spontaneous
        stimulus shown"""
        spontaneous_stimulus_blocks = get_spontaneous_block_indices(
            stimulus_blocks=(
                stimulus_presentations_table['stimulus_block'].values))
        res = []
        for spontaneous_block in spontaneous_stimulus_blocks:
            prev_stop_time = \
                stimulus_presentations_table[
                    stimulus_presentations_table['stimulus_block'] ==
                    spontaneous_block - 1]['stop_time'].max()
            prev_end_frame = \
                stimulus_presentations_table[
                    stimulus_presentations_table['stimulus_block'] ==
                    spontaneous_block - 1]['end_frame'].max()
            next_start_time = \
                stimulus_presentations_table[
                    stimulus_presentations_table['stimulus_block'] ==
                    spontaneous_block + 1]['start_time'].min()
            next_start_frame = \
                stimulus_presentations_table[
                    stimulus_presentations_table['stimulus_block'] ==
                    spontaneous_block + 1]['start_frame'].min()
            res.append({
                'duration': next_start_time - prev_stop_time,
                'start_time': prev_stop_time,
                'stop_time': next_start_time,
                'start_frame': prev_end_frame,
                'end_frame': next_start_frame,
                'stimulus_block': spontaneous_block,
                'stimulus_name': 'spontaneous'
            })
        res = pd.DataFrame(res)

        return res

    @classmethod
    def _add_fingerprint_stimulus(
            cls,
            stimulus_presentations: pd.DataFrame,
            stimulus_file: BehaviorStimulusFile,
            stimulus_timestamps: StimulusTimestamps
    ) -> pd.DataFrame:
        """Adds the fingerprint stimulus and the preceding gray screen to
        the stimulus presentations table

        Returns
        -------
        pd.DataFrame: stimulus presentations with gray screen + fingerprint
        movie added"""
        fingerprint_stimulus = cls._get_fingerprint_stimulus(
            stimulus_presentations=stimulus_presentations,
            stimulus_file=stimulus_file,
            stimulus_timestamps=stimulus_timestamps)
        stimulus_presentations = \
            pd.concat([stimulus_presentations, fingerprint_stimulus])
        spontaneous_stimulus = cls._get_spontaneous_stimulus(
            stimulus_presentations_table=stimulus_presentations)
        stimulus_presentations = \
            pd.concat([stimulus_presentations, spontaneous_stimulus])
        stimulus_presentations = \
            stimulus_presentations.sort_values('start_frame')

        # reset index to go from 0...end
        stimulus_presentations.index = pd.Index(
            np.arange(0, stimulus_presentations.shape[0]),
            name=stimulus_presentations.index.name,
            dtype=stimulus_presentations.index.dtype)
        return stimulus_presentations


def get_spontaneous_block_indices(
        stimulus_blocks: np.ndarray
) -> np.ndarray:
    """Gets the indexes where there is a gap in stimulus block

    Parameters
    ----------
    stimulus_blocks: Stimulus blocks in the stimulus presentations table
    """
    blocks = np.sort(np.unique(stimulus_blocks))
    block_diffs = np.diff(blocks)
    if (block_diffs > 2).any():
        raise RuntimeError(f'There should not be any stimulus block '
                           f'diffs greater than 2. The stimulus '
                           f'blocks are {blocks}')

    # i.e. if the current blocks are [0, 2], then block_diffs will
    # be [2], with a gap (== 2) at index 0, meaning that the spontaneous block
    # is at index 1
    block_indices = blocks[
        np.where(block_diffs == 2)[0]
    ] + 1
    return block_indices
