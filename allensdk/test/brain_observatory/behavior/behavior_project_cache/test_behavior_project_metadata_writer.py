import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache
from allensdk.brain_observatory.behavior.behavior_project_cache.external\
    .behavior_project_metadata_writer import \
    BehaviorProjectMetadataWriter
from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.session_type import \
    SessionType


class TestVBO:
    """Tests project tables for VBO"""
    @classmethod
    def setup_class(cls):

        test_dir = Path(__file__).parent / 'test_data' / 'vbo'

        # Note: these tables will need to be updated if the expected table
        # changes
        cls.expected_behavior_sessions_table = pd.read_csv(
            test_dir / 'behavior_session_table.csv')
        cls.expected_ophys_sessions_table = pd.read_csv(
            test_dir / 'ophys_session_table.csv')
        cls.expected_ophys_experiments_table = pd.read_csv(
            test_dir / 'ophys_experiment_table.csv')
        cls.expected_ophys_cells_table = pd.read_csv(
            test_dir / 'ophys_cells_table.csv')

        cls.session_type_map = (
            cls.expected_behavior_sessions_table
            .set_index('behavior_session_id')[['session_type']]
            .to_dict()['session_type'])

        cls.test_dir = tempfile.TemporaryDirectory()

        bpc = VisualBehaviorOphysProjectCache.from_lims(
            data_release_date=['2021-03-25', '2021-08-12'])
        cls.project_table_writer = BehaviorProjectMetadataWriter(
            behavior_project_cache=bpc,
            out_dir=cls.test_dir.name,
            project_name='',
            data_release_date='',
            overwrite_ok=True
        )

    def teardown_class(self):
        self.test_dir.cleanup()

    def _get_behavior_session(self, behavior_session_id, lims_db):
        if isinstance(behavior_session_id, BehaviorSessionId):
            behavior_session_id = behavior_session_id.value
        return BehaviorMetadata(
            date_of_acquisition=None,
            subject_metadata=None,
            behavior_session_id=BehaviorSessionId(behavior_session_id),
            equipment=None,
            stimulus_frame_rate=None,
            session_type=SessionType(
                self.session_type_map[behavior_session_id]),
            behavior_session_uuid=None
        )

    @pytest.mark.requires_bamboo
    def test_get_behavior_sessions_table(self):
        with patch.object(
            BehaviorMetadata, 'from_lims',
                wraps=self._get_behavior_session):
            self.project_table_writer._write_behavior_sessions(
                include_trial_metrics=False)
            obtained = pd.read_csv(Path(self.test_dir.name) /
                                   'behavior_session_table.csv')
            obtained = obtained.sort_values('behavior_session_id')\
                .reset_index(drop=True)
            expected = self.expected_behavior_sessions_table\
                .sort_values('behavior_session_id')\
                .reset_index(drop=True)
            pd.testing.assert_frame_equal(
                obtained, expected)

    @pytest.mark.requires_bamboo
    def test_get_ophys_sessions_table(self):
        with patch.object(
            BehaviorMetadata, 'from_lims',
                wraps=self._get_behavior_session):
            self.project_table_writer._write_ophys_sessions()
            obtained = pd.read_csv(Path(self.test_dir.name) /
                                   'ophys_session_table.csv')
            obtained = obtained.sort_values('ophys_session_id')\
                .reset_index(drop=True)
            expected = self.expected_ophys_sessions_table\
                .sort_values('ophys_session_id')\
                .reset_index(drop=True)
            pd.testing.assert_frame_equal(
                obtained, expected)

    @pytest.mark.requires_bamboo
    def test_get_ophys_experiments_table(self):
        with patch.object(
            BehaviorMetadata, 'from_lims',
                wraps=self._get_behavior_session):
            self.project_table_writer._write_ophys_experiments()
            obtained = pd.read_csv(Path(self.test_dir.name) /
                                   'ophys_experiment_table.csv')
            obtained = obtained.sort_values('ophys_experiment_id')\
                .reset_index(drop=True)
            expected = self.expected_ophys_experiments_table\
                .sort_values('ophys_experiment_id')\
                .reset_index(drop=True)
            pd.testing.assert_frame_equal(
                obtained.sort_index(axis=1), expected.sort_index(axis=1))

    @pytest.mark.requires_bamboo
    def test_get_ophys_cells_table(self):
        self.project_table_writer._write_ophys_cells()
        obtained = pd.read_csv(Path(self.test_dir.name) /
                               'ophys_cells_table.csv')
        pd.testing.assert_frame_equal(
            obtained, self.expected_ophys_cells_table)

    @pytest.mark.requires_bamboo
    def test_imaging_plane_group_only_mesoscope(self):
        """Tests that imaging plane group only applies to mesoscope"""
        with patch.object(
            BehaviorMetadata, 'from_lims',
                wraps=self._get_behavior_session):
            self.project_table_writer._write_ophys_sessions()
            self.project_table_writer._write_ophys_experiments()
        ophys_session_tbl = pd.read_csv(Path(self.test_dir.name) /
                                        'ophys_session_table.csv')
        ophys_experiment_tbl = pd.read_csv(Path(self.test_dir.name) /
                                           'ophys_experiment_table.csv')
        df = ophys_session_tbl.merge(ophys_experiment_tbl,
                                     on='ophys_session_id')

        assert (df[~df['equipment_name_x'].str.startswith('MESO')]
                ['imaging_plane_group_count'].isna().all())
        assert (df[~df['equipment_name_x'].str.startswith('MESO')]
                ['imaging_plane_group'].isna().all())

    @pytest.mark.requires_bamboo
    def test_imaging_plane_group_count_consistent(self):
        """Tests that imaging plane group count in ophys sessions table is
        consistent with the number of imaging plane groups in experiment
        table"""
        with patch.object(
            BehaviorMetadata, 'from_lims',
                wraps=self._get_behavior_session):
            self.project_table_writer._write_ophys_sessions()
            self.project_table_writer._write_ophys_experiments()
        ophys_session_tbl = pd.read_csv(Path(self.test_dir.name) /
                                        'ophys_session_table.csv')
        ophys_experiment_tbl = pd.read_csv(Path(self.test_dir.name) /
                                           'ophys_experiment_table.csv')
        df = ophys_session_tbl.merge(ophys_experiment_tbl,
                                     on='ophys_session_id')

        imaging_plane_group_count = (
            df[~df['imaging_plane_group'].isna()]
            .groupby('ophys_session_id')['imaging_plane_group'].nunique()
            .reset_index()
            .rename(
                columns={'imaging_plane_group': 'imaging_plane_group_count'})
        )

        ophys_session_tbl = ophys_session_tbl.merge(
            imaging_plane_group_count,
            on='ophys_session_id',
            suffixes=('_from_lims', '_recalculated'),
            how='left'
        )
        assert (
            (ophys_session_tbl['imaging_plane_group_count_from_lims']
             [~ophys_session_tbl['imaging_plane_group_count_from_lims'].isna()]
             ==
             ophys_session_tbl['imaging_plane_group_count_recalculated']
             [~ophys_session_tbl['imaging_plane_group_count_recalculated']
             .isna()])
            .all()
        )
