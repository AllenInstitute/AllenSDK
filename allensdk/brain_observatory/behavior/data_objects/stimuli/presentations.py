import collections
from pathlib import Path
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.brain_observatory.behavior.data_objects.stimuli\
    .fingerprint_stimulus import \
    FingerprintStimulus
from allensdk.core import DataObject
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.core import \
    NwbReadableInterface
from allensdk.brain_observatory.behavior.data_files.stimulus_file import \
    StimulusFileReadableInterface, MalformedStimulusFileError
from allensdk.core import \
    NwbWritableInterface
from allensdk.brain_observatory.behavior.stimulus_processing import \
    get_stimulus_presentations, get_stimulus_metadata, is_change_event, \
    get_flashes_since_change, fix_omitted_end_frame
from allensdk.brain_observatory.nwb import \
    create_stimulus_presentation_time_interval
from allensdk.internal.brain_observatory.mouse import Mouse


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
                if series.dtype.name == 'boolean':
                    # Fixing an issue in which a bool column contains
                    # nans, which get coerced to True in pynwb
                    # Float maintains the nan values, while bool does not
                    # These will be coerced to boolean when reading
                    cleaned_table[colname] = series.astype(float)
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

        # coercing bool columns with nans to boolean. "Boolean" dtype
        # allows null values, while "bool" does not (see comment in to_nwb)
        table = table.astype({
            c: 'boolean' for c in table.select_dtypes(include='float')
            if set(table[c][~table[c].isna()].unique()).issubset({1, 0})
               # These columns should not be coerced to boolean
               and not c.endswith('_index')
        })
        table = table.sort_values(by=["start_time"])

        table = table.reset_index(drop=True)
        table.index = table.index.astype('int64')

        if add_is_change:
            table['is_change'] = is_change_event(stimulus_presentations=table)
            table['flashes_since_change'] = \
                get_flashes_since_change(stimulus_presentations=table)
        return Presentations(presentations=table, column_list=column_list)

    @classmethod
    def from_stimulus_file(
            cls,
            stimulus_file: BehaviorStimulusFile,
            stimulus_timestamps: StimulusTimestamps,
            behavior_session_id: int,
            limit_to_images: Optional[List] = None,
            column_list: Optional[List[str]] = None,
            fill_omitted_values=True
    ) -> "Presentations":
        """Get stimulus presentation data.

        :param stimulus_file
        :param limit_to_images
            Only return images given by these image names
        :param stimulus_timestamps
        :param behavior_session_id
            LIMS id of behavior session
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
        stim_pres_df['flashes_since_change'] = get_flashes_since_change(
            stimulus_presentations=stim_pres_df)

        # Sort columns then drop columns which contain only all NaN values
        stim_pres_df = \
            stim_pres_df[sorted(stim_pres_df)].dropna(axis=1, how='all')
        if limit_to_images is not None:
            stim_pres_df = \
                stim_pres_df[stim_pres_df['image_name'].isin(limit_to_images)]
            stim_pres_df.index = pd.Int64Index(
                range(stim_pres_df.shape[0]), name=stim_pres_df.index.name)

        stim_pres_df['stimulus_block'] = 0
        # Match the Ecephys VBN stimulus name convention.
        try:
            stim_pres_df['stimulus_name'] = Path(
                stimulus_file.stimuli['images']['image_set']).\
                stem.split('.')[0]
        except KeyError:
            # if we can't find the images key in the stimuli, check for the
            # name ``grating`` as the stimulus. If not add generic
            # ``behavior``.
            if 'grating' in stimulus_file.stimuli.keys():
                stim_pres_df['stimulus_name'] = 'grating'
            else:
                stim_pres_df['stimulus_name'] = 'behavior'

        stim_pres_df = fix_omitted_end_frame(stim_pres_df)

        cls._add_is_image_novel(stimulus_presentations=stim_pres_df,
                                behavior_session_id=behavior_session_id)

        has_fingerprint_stimulus = \
            'fingerprint' in stimulus_file.data['items']['behavior']['items']
        if has_fingerprint_stimulus:
            stim_pres_df = cls._add_fingerprint_stimulus(
                stimulus_presentations=stim_pres_df,
                stimulus_file=stimulus_file,
                stimulus_timestamps=stimulus_timestamps
            )
        stim_pres_df = cls._postprocess(
            presentations=stim_pres_df,
            fill_omitted_values=fill_omitted_values,
            coerce_bool_to_boolean=True
        )
        return Presentations(presentations=stim_pres_df,
                             column_list=column_list)

    @classmethod
    def from_path(cls,
                  path: Union[str, Path],
                  behavior_session_id: int,
                  exclude_columns: Optional[List[str]] = None,
                  columns_to_rename: Optional[Dict[str, str]] = None,
                  sort_columns: bool = True
                  ) -> "Presentations":
        """
        Reads the table directly from a precomputed csv

        Parameters
        -----------
        path: Path to load table from
        behavior_session_id
            LIMS behavior session id
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
        cls._add_is_image_novel(stimulus_presentations=df,
                                behavior_session_id=behavior_session_id)
        exclude_columns = exclude_columns if exclude_columns else []
        df = df[[c for c in df if c not in exclude_columns]]
        df = cls._postprocess(presentations=df,
                              fill_omitted_values=False,
                              coerce_bool_to_boolean=True)
        return Presentations(presentations=df,
                             columns_to_rename=columns_to_rename,
                             sort_columns=sort_columns)

    @classmethod
    def _get_is_image_novel(
            cls,
            image_names: List[str],
            behavior_session_id: int,
    ) -> Dict[str, bool]:
        """
        Returns whether each image in `image_names` is novel for the mouse

        Parameters
        ----------
        image_names:
            List of image names
        behavior_session_id
            LIMS behavior session id

        Returns
        -------
        Dict mapping image name to is_novel
        """
        mouse = Mouse.from_behavior_session_id(
            behavior_session_id=behavior_session_id)
        prior_images_shown = mouse.get_images_shown(
            up_to_behavior_session_id=behavior_session_id)

        image_names = set([
            x for x in image_names if x != 'omitted' and type(x) is str])
        is_novel = {
            f'{image_name}': image_name not in prior_images_shown
            for image_name in image_names}
        return is_novel

    @classmethod
    def _add_is_image_novel(
            cls,
            stimulus_presentations: pd.DataFrame,
            behavior_session_id: int
    ):
        """Adds a column 'is_image_novel' to `stimulus_presentations`

        Parameters
        ----------
        stimulus_presentations: stimulus presentations table
        behavior_session_id: LIMS id of behavior session

        """
        stimulus_presentations['is_image_novel'] = \
            stimulus_presentations['image_name']\
            .map(cls._get_is_image_novel(
                image_names=stimulus_presentations['image_name'].tolist(),
                behavior_session_id=behavior_session_id
            ))

    @classmethod
    def _postprocess(cls, presentations: pd.DataFrame,
                     fill_omitted_values=True,
                     coerce_bool_to_boolean=True,
                     omitted_time_duration: float = 0.25) \
            -> pd.DataFrame:
        """
        Applies further processing to `presentations`

        Parameters
        ----------
        presentations
            Presentations df
        fill_omitted_values
            Whether to fill stop time and duration for omitted flashes
        coerce_bool_to_boolean
            Whether to coerce columns of "Object" dtype that are truly bool
            to nullable "boolean" dtype
        omitted_time_duration
            Amount of time a stimuli is omitted for in seconds"""
        df = presentations
        if fill_omitted_values:
            cls._fill_missing_values_for_omitted_flashes(
                df=df, omitted_time_duration=omitted_time_duration)
        if coerce_bool_to_boolean:
            df = df.astype({
                c: 'boolean' for c in df.select_dtypes('O')
                if set(df[c][~df[c].isna()].unique()).issubset({True, False})})
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
        omitted = df['omitted'].fillna(False)
        df.loc[omitted, 'stop_time'] = \
            df.loc[omitted, 'start_time'] + omitted_time_duration
        df.loc[omitted, 'duration'] = omitted_time_duration
        return df

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
        try:
            fingerprint_stimulus = FingerprintStimulus.from_stimulus_file(
                stimulus_presentations=stimulus_presentations,
                stimulus_file=stimulus_file,
                stimulus_timestamps=stimulus_timestamps)
        except MalformedStimulusFileError:
            return stimulus_presentations

        stimulus_presentations = \
            pd.concat([stimulus_presentations, fingerprint_stimulus.table])
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
    """Gets the indices where there is a gap in stimulus block. This is
    where spontaneous blocks are.
    Example: stimulus blocks are [0, 2, 3]. There is a spontaneous block at 1.

    Parameters
    ----------
    stimulus_blocks: Stimulus blocks in the stimulus presentations table

    Notes
    -----
    This doesn't support a spontaneous block appearing at the beginning or
    end of a session

    Returns
    -------
    np.array: spontaneous stimulus blocks
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
