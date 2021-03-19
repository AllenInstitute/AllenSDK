import argparse
import json
import logging
import os
from typing import Union

import pandas as pd

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .experiments_table import \
    ExperimentsTable
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .ophys_sessions_table import \
    BehaviorOphysSessionsTable
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .sessions_table import \
    SessionsTable

SESSION_SUPPRESS = (
    'donor_id',
    'foraging_id',
    'session_name',
    'specimen_id'
)
OPHYS_EXPERIMENTS_SUPPRESS = SESSION_SUPPRESS + (
    'container_workflow_state',
    'behavior_session_uuid',
    'experiment_workflow_state',
    'published_at',
    'isi_experiment_id'
)
OUTPUT_METADATA_FILENAMES = {
    'behavior_session_table': 'behavior_session_table.csv',
    'ophys_session_table': 'ophys_session_table.csv',
    'ophys_experiment_table': 'ophys_experiment_table.csv'
}


class BehaviorProjectMetadataWriter:
    """Class to write project-level metadata to csv"""

    def __init__(self, behavior_project_cache: VisualBehaviorOphysProjectCache,
                 out_dir: str, project_name: str, data_release_date: str):
        self._behavior_project_cache = behavior_project_cache
        self._out_dir = out_dir
        self._project_name = project_name
        self._data_release_date = data_release_date
        self._logger = logging.getLogger(self.__class__.__name__)

        self._release_behavior_only_nwb = self._behavior_project_cache \
            .fetch_api.get_release_files(file_type='BehaviorNwb')
        self._release_behavior_with_ophys_nwb = self._behavior_project_cache \
            .fetch_api.get_release_files(file_type='BehaviorOphysNwb')

    @property
    def release_behavior_only_nwb(self):
        """Returns all release behavior only nwb"""
        return self._release_behavior_only_nwb

    @property
    def release_behavior_with_ophys_nwb(self):
        """Returns all release behavior only nwb"""
        return self._release_behavior_with_ophys_nwb

    def write_metadata(self):
        """Writes metadata to csv"""
        os.makedirs(self._out_dir, exist_ok=True)

        self._write_behavior_sessions()
        self._write_ophys_sessions()
        self._write_ophys_experiments()

        self._write_manifest()

    def _write_behavior_sessions(self, suppress=SESSION_SUPPRESS,
                                 output_filename=OUTPUT_METADATA_FILENAMES[
                                     'behavior_session_table']):
        behavior_sessions = self._behavior_project_cache. \
            get_behavior_session_table(suppress=suppress,
                                       as_df=False)
        behavior_sessions = self._get_release_table(table=behavior_sessions)
        self._write_metadata_table(df=behavior_sessions,
                                   filename=output_filename)

    def _write_ophys_sessions(self, suppress=SESSION_SUPPRESS,
                              output_filename=OUTPUT_METADATA_FILENAMES[
                                  'ophys_session_table'
                              ]):
        ophys_sessions = self._behavior_project_cache. \
            get_session_table(suppress=suppress, as_df=False)
        ophys_sessions = self._get_release_table(table=ophys_sessions)
        self._write_metadata_table(df=ophys_sessions,
                                   filename=output_filename)

    def _write_ophys_experiments(self, suppress=OPHYS_EXPERIMENTS_SUPPRESS,
                                 output_filename=OUTPUT_METADATA_FILENAMES[
                                     'ophys_experiment_table'
                                 ]):
        ophys_experiments = self._behavior_project_cache.get_experiment_table(
            suppress=suppress, as_df=False)
        ophys_experiments = self._get_release_table(table=ophys_experiments)
        self._write_metadata_table(df=ophys_experiments,
                                   filename=output_filename)

    def _get_release_table(self,
                           table: Union[
                               SessionsTable,
                               BehaviorOphysSessionsTable,
                               ExperimentsTable]) -> pd.DataFrame:
        """Takes as input an entire project-level table and filters it to
        include records which we are releasing data for

        Parameters
        ----------
        table
            The project table to filter

        Returns
        --------
        The filtered dataframe
        """
        if isinstance(table, SessionsTable):
            release_table = table.table.merge(self._release_behavior_only_nwb,
                                              left_index=True,
                                              right_index=True,
                                              how='left')
        elif isinstance(table, BehaviorOphysSessionsTable):
            release_table = table.table
        elif isinstance(table, ExperimentsTable):
            release_table = table.table.merge(
                self._release_behavior_with_ophys_nwb
                    .drop('behavior_session_id', axis=1),
                left_index=True,
                right_index=True,
                how='left')
        else:
            raise ValueError(f'Bad table {type(table)}')

        return release_table

    def _write_metadata_table(self, df: pd.DataFrame, filename: str):
        """
        Writes file to csv

        Parameters
        ----------
        df
            The dataframe to write
        filename
            Filename to save as
        """
        filepath = os.path.join(self._out_dir, filename)
        self._logger.info(f'Writing {filepath}')

        df = df.reset_index()
        df.to_csv(filepath, index=False)

        self._logger.info('Writing successful')

    def _write_manifest(self):
        data_files = \
            self._release_behavior_only_nwb['isilon_filepath'].to_list() + \
            self._release_behavior_with_ophys_nwb['isilon_filepath'].to_list()
        filenames = OUTPUT_METADATA_FILENAMES.values()

        def get_abs_path(filename):
            return os.path.abspath(os.path.join(self._out_dir, filename))

        metadata_files = [get_abs_path(f) for f in filenames]
        data_pipeline = [{
            'name': 'AllenSDK',
            'version': allensdk.__version__,
            'comment': 'AllenSDK version used to produce data NWB and '
                       'metadata CSV files for this release'
        }]

        manifest = {
            'data_files': data_files,
            'metadata_files': metadata_files,
            'data_pipeline': data_pipeline,
            'project_name': self._project_name,
            'data_release_date': self._data_release_date
        }

        save_path = os.path.join(self._out_dir, 'manifest.json')
        with open(save_path, 'w') as f:
            f.write(json.dumps(manifest, indent=4))


def main():
    parser = argparse.ArgumentParser(description='Write project metadata to '
                                                 'csvs')
    parser.add_argument('-out_dir', help='directory to save csvs',
                        required=True)
    parser.add_argument('-project_name', help='project name', required=True)
    parser.add_argument('-data_release_date', help='Project release date. '
                                                   'Ie 2021-03-25',
                        required=True)
    args = parser.parse_args()

    bpc = VisualBehaviorOphysProjectCache.from_lims(
        data_release_date=args.data_release_date)
    bpmw = BehaviorProjectMetadataWriter(
        behavior_project_cache=bpc,
        out_dir=args.out_dir,
        project_name=args.project_name,
        data_release_date=args.data_release_date)
    bpmw.write_metadata()


if __name__ == '__main__':
    main()
