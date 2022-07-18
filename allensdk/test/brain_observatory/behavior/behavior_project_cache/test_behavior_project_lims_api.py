from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io import BehaviorProjectLimsApi  # noqa: E501

from allensdk.test_utilities.custom_comparators import (
    WhitespaceStrippedString)


class MockQueryEngine:
    def __init__(self, **kwargs):
        pass

    def select(self, query):
        return query

    def fetchall(self, query):
        return query

    def stream(self, endpoint):
        return endpoint


@pytest.fixture
def MockBehaviorProjectLimsApi():
    return BehaviorProjectLimsApi(MockQueryEngine(), MockQueryEngine(),
                                  MockQueryEngine())


@pytest.mark.parametrize(
    "line,expected", [
        ("reporter", WhitespaceStrippedString(
            """-- -- begin getting reporter line from donors -- --
            SELECT ARRAY_AGG (g.name) AS reporter_line, d.id AS donor_id
            FROM donors d
            LEFT JOIN donors_genotypes dg ON dg.donor_id=d.id
            LEFT JOIN genotypes g ON g.id=dg.genotype_id
            LEFT JOIN genotype_types gt ON gt.id=g.genotype_type_id
            WHERE gt.name='reporter'
            GROUP BY d.id
            -- -- end getting reporter line from donors -- --""")),
        ("driver", WhitespaceStrippedString(
            """-- -- begin getting driver line from donors -- --
            SELECT ARRAY_AGG (g.name) AS driver_line, d.id AS donor_id
            FROM donors d
            LEFT JOIN donors_genotypes dg ON dg.donor_id=d.id
            LEFT JOIN genotypes g ON g.id=dg.genotype_id
            LEFT JOIN genotype_types gt ON gt.id=g.genotype_type_id
            WHERE gt.name='driver'
            GROUP BY d.id
            -- -- end getting driver line from donors -- --"""))
    ]
)
def test_build_line_from_donor_query(line, expected,
                                     MockBehaviorProjectLimsApi):
    mbp_api = MockBehaviorProjectLimsApi
    assert expected == mbp_api._build_line_from_donor_query(line=line)


class TestProjectTablesAll:
    """Tests for passing passed_only=False to project tables"""
    @classmethod
    def setup_class(cls):
        test_dir = Path(__file__).parent / 'test_data' / 'vbo'

        # Note: these tables will need to be updated if the expected table
        # changes
        cls.release_behavior_sessions_table = pd.read_csv(
            test_dir / 'behavior_session_table.csv')
        cls.release_ophys_sessions_table = pd.read_csv(
            test_dir / 'ophys_session_table.csv')
        cls.release_ophys_experiments_table = pd.read_csv(
            test_dir / 'ophys_experiment_table.csv')
        cls.release_ophys_cells_table = pd.read_csv(
            test_dir / 'ophys_cells_table.csv')

        cls.lims_api = BehaviorProjectLimsApi.default(
            passed_only=False
        )

    def _get_session_type(self, behavior_session_id, db_conn):
        """
        Note: mocking this because getting session type from pkl file is
        expensive
        """
        return {
            'behavior_session_id': behavior_session_id,
            'session_type': 'foo'}

    @pytest.mark.requires_bamboo
    def test_all_behavior_sessions(self):
        """Tests that when passed_only=False, that more sessions are returned
        than in the release table"""
        with patch('allensdk.brain_observatory.'
                   'behavior.behavior_project_cache.'
                   'project_apis.data_io.behavior_project_lims_api.'
                   '_get_session_type_from_pkl_file',
                   wraps=self._get_session_type):
            obtained = self.lims_api.get_behavior_session_table()

            # Make sure ids returned are superset of release ids
            assert len(set(obtained.index).intersection(
                self.release_behavior_sessions_table['behavior_session_id'])) \
                == \
                self.release_behavior_sessions_table['behavior_session_id']\
                .nunique()
            assert obtained.shape[0] > \
                   self.release_behavior_sessions_table.shape[0]

    @pytest.mark.requires_bamboo
    def test_all_ophys_sessions(self):
        """Tests that when passed_only=False, that more sessions are returned
        than in the release table"""
        with patch('allensdk.brain_observatory.'
                   'behavior.behavior_project_cache.'
                   'project_apis.data_io.behavior_project_lims_api.'
                   '_get_session_type_from_pkl_file',
                   wraps=self._get_session_type):
            obtained = self.lims_api.get_ophys_session_table()

            # Make sure ids returned are superset of release ids
            assert len(set(obtained.index).intersection(
                self.release_ophys_sessions_table['ophys_session_id'])) \
                == \
                self.release_ophys_sessions_table['ophys_session_id']\
                .nunique()
            assert obtained.shape[0] > \
                   self.release_ophys_sessions_table.shape[0]

    @pytest.mark.requires_bamboo
    def test_all_ophys_experiments(self):
        """Tests that when passed_only=False, that more experiments are
        returned than in the release table"""
        with patch('allensdk.brain_observatory.'
                   'behavior.behavior_project_cache.'
                   'project_apis.data_io.behavior_project_lims_api.'
                   '_get_session_type_from_pkl_file',
                   wraps=self._get_session_type):
            obtained = self.lims_api.get_ophys_experiment_table()

            # Make sure ids returned are superset of release ids
            assert len(set(obtained.index).intersection(
                self.release_ophys_experiments_table['ophys_experiment_id'])) \
                == \
                self.release_ophys_experiments_table['ophys_experiment_id']\
                .nunique()
            assert obtained.shape[0] > \
                   self.release_ophys_experiments_table.shape[0]

    @pytest.mark.requires_bamboo
    def test_all_cells(self):
        """Tests that when passed_only=False, that more experiments are
        returned than in the release table"""
        obtained = self.lims_api.get_ophys_cells_table()

        # Make sure ids returned are superset of release ids
        assert len(set(obtained['ophys_experiment_id']).intersection(
            self.release_ophys_cells_table['ophys_experiment_id'])) \
            == \
            self.release_ophys_cells_table['ophys_experiment_id']\
            .nunique()
        assert obtained.shape[0] > \
               self.release_ophys_cells_table.shape[0]
