import argparse
import json
import logging
import os
import warnings
from typing import List, Union

import pandas as pd

import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache

#########
# These columns should be dropped from external-facing metadata
#########
SESSION_SUPPRESS = (
    'donor_id',
    'foraging_id',
    'session_name',
    'specimen_id'
)
OPHYS_EXPERIMENTS_SUPPRESS = SESSION_SUPPRESS + (
    'behavior_session_uuid',
    'published_at',
    'isi_experiment_id'
)
OPHYS_EXPERIMENTS_SUPPRESS_FINAL = [
    'container_workflow_state',
    'experiment_workflow_state']
#########

OUTPUT_METADATA_FILENAMES = {
    'behavior_session_table': 'behavior_session_table.csv',
    'ophys_session_table': 'ophys_session_table.csv',
    'ophys_experiment_table': 'ophys_experiment_table.csv',
    'ophys_cells_table': 'ophys_cells_table.csv'
}


class BehaviorProjectMetadataWriter:
    """Class to write project-level metadata to csv"""

    def __init__(self, behavior_project_cache: VisualBehaviorOphysProjectCache,
                 out_dir: str, project_name: str,
                 data_release_date: Union[str, List[str]],
                 overwrite_ok=False,
                 n_workers=1):
        """

        Parameters
        ----------
        behavior_project_cache
            Cache object to use for fetching
        out_dir
            Where to write tables
        project_name
            Project name in LIMS to use for querying data to be released
        data_release_date
            Date data is released. Needed to filter results to only released
                data
        overwrite_ok
            Whether to raise an exception if output file alread exists
        n_workers
            Number of parallel processes to use for i.e reading pkl files
        """

        self._behavior_project_cache = behavior_project_cache
        self._out_dir = out_dir
        self._project_name = project_name
        self._data_release_date = data_release_date
        self._overwrite_ok = overwrite_ok
        self._n_workers = n_workers

        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)

        self._release_behavior_only_nwb = self._behavior_project_cache \
            .fetch_api.get_release_files(file_type='BehaviorNwb')
        self._release_behavior_with_ophys_nwb = self._behavior_project_cache \
            .fetch_api.get_release_files(file_type='BehaviorOphysNwb')

    def write_metadata(self):
        """Writes metadata to csv"""
        os.makedirs(self._out_dir, exist_ok=True)

        self._logger.info('Writing behavior sessions table')
        self._write_behavior_sessions()
        self._logger.info('Writing ophys sessions table')
        self._write_ophys_sessions()
        self._logger.info('Writing ophys experiments table')
        self._write_ophys_experiments()
        self._logger.info('Writing ophys cells table')
        self._write_ophys_cells()

        self._write_manifest()

    def _write_behavior_sessions(
            self,
            suppress=SESSION_SUPPRESS,
            output_filename=OUTPUT_METADATA_FILENAMES[
                'behavior_session_table'],
            include_trial_metrics: bool = True
    ):
        behavior_sessions = self._behavior_project_cache. \
            get_behavior_session_table(
                suppress=suppress,
                as_df=True,
                n_workers=self._n_workers,
                include_trial_metrics=include_trial_metrics)

        # Add release files
        behavior_sessions = behavior_sessions \
            .merge(self._release_behavior_only_nwb,
                   left_index=True,
                   right_index=True,
                   how='left')
        if "file_id" in behavior_sessions.columns:
            if behavior_sessions["file_id"].isnull().values.any():
                msg = (f"{output_filename} field `file_id` contains missing "
                       "values and pandas.to_csv() converts it to float")
                warnings.warn(msg)
        self._write_metadata_table(df=behavior_sessions,
                                   filename=output_filename)

    def _write_ophys_cells(self,
                           output_filename=OUTPUT_METADATA_FILENAMES[
                                'ophys_cells_table']):
        ophys_cells = self._behavior_project_cache. \
            get_ophys_cells_table()
        self._write_metadata_table(df=ophys_cells,
                                   filename=output_filename)

    def _write_ophys_sessions(self, suppress=SESSION_SUPPRESS,
                              output_filename=OUTPUT_METADATA_FILENAMES[
                                  'ophys_session_table'
                              ]):
        ophys_sessions = self._behavior_project_cache. \
            get_ophys_session_table(suppress=suppress, as_df=True,
                                    n_workers=self._n_workers)
        self._write_metadata_table(df=ophys_sessions,
                                   filename=output_filename)

    def _write_ophys_experiments(self, suppress=OPHYS_EXPERIMENTS_SUPPRESS,
                                 output_filename=OUTPUT_METADATA_FILENAMES[
                                     'ophys_experiment_table'
                                 ]):
        ophys_experiments = \
                self._behavior_project_cache.get_ophys_experiment_table(
                        suppress=suppress, as_df=True,
                        n_workers=self._n_workers)

        # Add release files
        ophys_experiments = ophys_experiments.merge(
            self._release_behavior_with_ophys_nwb
                .drop('behavior_session_id', axis=1),
            left_index=True,
            right_index=True,
            how='left')

        # users don't need to see these
        ophys_experiments.drop(
                labels=OPHYS_EXPERIMENTS_SUPPRESS_FINAL,
                inplace=True,
                axis=1)

        self._write_metadata_table(df=ophys_experiments,
                                   filename=output_filename)

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
        self._pre_file_write(filepath=filepath)

        self._logger.info(f'Writing {filepath}')

        df = df.reset_index()
        df.to_csv(filepath, index=False)

    def _write_manifest(self):
        def get_abs_path(filename):
            return os.path.abspath(os.path.join(self._out_dir, filename))

        metadata_filenames = OUTPUT_METADATA_FILENAMES.values()
        metadata_files = [get_abs_path(f) for f in metadata_filenames]
        data_pipeline = [{
            'name': 'AllenSDK',
            'version': allensdk.__version__,
            'comment': 'AllenSDK version used to produce data NWB and '
                       'metadata CSV files for this release'
        }]

        manifest = {
            'metadata_files': metadata_files,
            'data_pipeline_metadata': data_pipeline,
            'project_name': self._project_name,
        }

        save_path = os.path.join(self._out_dir, 'manifest.json')
        self._pre_file_write(filepath=save_path)

        with open(save_path, 'w') as f:
            f.write(json.dumps(manifest, indent=4))

    def _pre_file_write(self, filepath: str):
        """Checks if file exists at filepath. If so, and overwrite_ok is False,
        raises an exception"""
        if os.path.exists(filepath):
            if self._overwrite_ok:
                pass
            else:
                raise RuntimeError(f'{filepath} already exists. In order '
                                   f'to overwrite this file, pass the '
                                   f'--overwrite_ok flag')


def main():
    parser = argparse.ArgumentParser(description='Write project metadata to '
                                                 'csvs')
    parser.add_argument('--out_dir', help='directory to save csvs',
                        required=True)
    parser.add_argument('--project_name', help='project name', required=True)
    parser.add_argument('--data_release_date', help='Project release date. '
                        'Ie 2021-03-25',
                        required=True,
                        nargs="+")
    parser.add_argument('--overwrite_ok', help='Whether to allow overwriting '
                                               'existing output files',
                        dest='overwrite_ok', action='store_true')
    parser.add_argument('--n_workers', help='Number of processes to use for '
                                            'i.e reading from pkl files',
                        default=1, type=int)
    args = parser.parse_args()

    bpc = VisualBehaviorOphysProjectCache.from_lims(
        data_release_date=args.data_release_date)
    bpmw = BehaviorProjectMetadataWriter(
        behavior_project_cache=bpc,
        out_dir=args.out_dir,
        project_name=args.project_name,
        data_release_date=args.data_release_date,
        overwrite_ok=args.overwrite_ok,
        n_workers=args.n_workers
    )
    bpmw.write_metadata()


if __name__ == '__main__':
    main()
