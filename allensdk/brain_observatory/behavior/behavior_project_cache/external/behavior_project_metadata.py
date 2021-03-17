import argparse
import os
from typing import Union

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    BehaviorProjectCache
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .experiments_table import \
    ExperimentsTable
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .ophys_sessions_table import \
    BehaviorOphysSessionsTable
from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .sessions_table import \
    SessionsTable


class BehaviorProjectMetadataWriter:
    """Class to write project-level metadata to csv"""

    def __init__(self, behavior_project_cache: BehaviorProjectCache):
        self._behavior_project_cache = behavior_project_cache

    def write_metadata(self, out_dir: str):
        """Writes metadata to csv

        Parameters
        ----------
        out_dir
            Output directory
        """
        os.makedirs(out_dir, exist_ok=True)

        behavior_suppress = [
            'donor_id',
            'foraging_id'
        ]
        ophys_suppress = [
            'session_name',
            'donor_id',
            'specimen_id'
        ]
        ophys_experiments_suppress = ophys_suppress + [
            'container_workflow_state',
            'behavior_session_uuid',
            'experiment_workflow_state',
            'published_at',
        ]
        self._get_behavior_sessions(
            suppress=behavior_suppress).reset_index().to_csv(
            os.path.join(out_dir, 'behavior_session_table.csv'))
        self._get_behavior_ophys_sessions(
            suppress=ophys_suppress).reset_index().to_csv(
            os.path.join(out_dir, 'ophys_session_table.csv'))
        self._get_behavior_ophys_experiments(
            suppress=ophys_experiments_suppress).reset_index().to_csv(
            os.path.join(out_dir, 'ophys_experiment_table.csv'))

    def _get_behavior_sessions(self, suppress=None):
        behavior_sessions = self._behavior_project_cache. \
            get_behavior_session_table(suppress=suppress, as_df=False)
        return self._get_release_table(table=behavior_sessions)

    def _get_behavior_ophys_sessions(self, suppress=None):
        ophys_sessions = self._behavior_project_cache. \
            get_session_table(suppress=suppress, as_df=False)
        return self._get_release_table(table=ophys_sessions)

    def _get_behavior_ophys_experiments(self, suppress=None):
        ophys_experiments = self._behavior_project_cache.get_experiment_table(
            suppress=suppress, as_df=False)
        return self._get_release_table(table=ophys_experiments)

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
            release_files = self._get_release_files(
                file_type='BehaviorNwb')
        elif isinstance(table, BehaviorOphysSessionsTable) or \
                isinstance(table, ExperimentsTable):
            release_files = self._get_release_files(
                file_type='BehaviorOphysNwb')
            if isinstance(table, BehaviorOphysSessionsTable):
                # ophys sessions are different because the nwb files for ophys
                # sessions are at the experiment level.
                # We don't want to associate these sessions with nwb files
                ophys_session_ids = \
                    self._get_ophys_sessions_from_ophys_experiments(
                        ophys_experiment_ids=release_files.index)
                return table.table[table.table.index.isin(ophys_session_ids)]
        else:
            raise ValueError(f'Bad table {type(table)}')

        return table.table.merge(release_files, left_index=True,
                                 right_index=True)

    def _get_release_files(self, file_type='BehaviorNwb') -> pd.DataFrame:
        """Gets the release nwb files.

        Parameters
        ----------
        file_type
            NWB files to return ('BehaviorNwb', 'BehaviorOphysNwb')

        Returns
        ---------
        Dataframe of release files and file metadata
        """
        if file_type not in ('BehaviorNwb', 'BehaviorOphysNwb'):
            raise ValueError(f'cannot retrieve file type {file_type}')

        if file_type == 'BehaviorNwb':
            attachable_id_alias = 'behavior_session_id'
        else:
            attachable_id_alias = 'ophys_experiment_id'

        query = f'''
            SELECT attachable_id as {attachable_id_alias}, id as file_id, 
            filename, storage_directory
            FROM well_known_files 
            WHERE published_at IS NOT NULL AND 
                well_known_file_type_id IN (
                    SELECT id 
                    FROM well_known_file_types 
                    WHERE name = '{file_type}'
                );
        '''
        res = self._behavior_project_cache.fetch_api.lims_engine.select(query)
        res['isilon_filepath'] = res['storage_directory'] \
            .str.cat(res['filename'])
        res = res.drop(['filename', 'storage_directory'], axis=1)
        return res.set_index(attachable_id_alias)

    def _get_ophys_sessions_from_ophys_experiments(
            self, ophys_experiment_ids: pd.Series):
        session_query = self._behavior_project_cache.fetch_api. \
            build_in_list_selector_query(
            "oe.id", ophys_experiment_ids.to_list())

        query = f'''
            SELECT os.id as ophys_session_id
            FROM ophys_sessions os
            JOIN ophys_experiments oe on oe.ophys_session_id = os.id
            {session_query}
        '''
        res = self._behavior_project_cache.fetch_api.lims_engine.select(query)
        return res['ophys_session_id']


def main():
    parser = argparse.ArgumentParser(description='Write project metadata to '
                                                 'csvs')
    parser.add_argument('-out_dir', help='directory to save csvs',
                        required=True)
    args = parser.parse_args()

    bpc = BehaviorProjectCache.from_lims()
    bpmw = BehaviorProjectMetadataWriter(behavior_project_cache=bpc)
    bpmw.write_metadata(out_dir=args.out_dir)


if __name__ == '__main__':
    main()
