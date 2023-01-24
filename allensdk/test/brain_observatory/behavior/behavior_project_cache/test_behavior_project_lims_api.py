import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache
from allensdk.brain_observatory.behavior.behavior_project_cache.external \
    .behavior_project_metadata_writer import \
    SESSION_SUPPRESS, OPHYS_EXPERIMENTS_SUPPRESS, \
    OPHYS_EXPERIMENTS_SUPPRESS_FINAL
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io import \
    BehaviorProjectLimsApi, BehaviorProjectCloudApi  # noqa: E501
from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.session_type import \
    SessionType

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

    def _get_behavior_session(self, behavior_session_id, lims_db):
        if isinstance(behavior_session_id, BehaviorSessionId):
            behavior_session_id = behavior_session_id.value
        return BehaviorMetadata(
            date_of_acquisition=None,
            subject_metadata=None,
            behavior_session_id=BehaviorSessionId(behavior_session_id),
            equipment=None,
            stimulus_frame_rate=None,
            session_type=SessionType('foo'),
            behavior_session_uuid=None
        )

    @pytest.mark.requires_bamboo
    def test_all_behavior_sessions(self):
        """Tests that when passed_only=False, that more sessions are returned
        than in the release table"""
        with patch.object(
                BehaviorMetadata,
                'from_lims', wraps=self._get_behavior_session):
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
        with patch.object(
                BehaviorMetadata,
                'from_lims', wraps=self._get_behavior_session):
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
        with patch.object(
                BehaviorMetadata,
                'from_lims', wraps=self._get_behavior_session):
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


class TestLimsCloudConsistency:
    """Tests for checking consistency between tables as returned by cloud
    vs those returned internally by lims"""
    @classmethod
    def setup_class(cls):
        tempdir = tempfile.TemporaryDirectory()

        cls.test_dir = Path(__file__).parent / 'test_data' / 'vbo'

        behavior_sessions_table = pd.read_csv(
            cls.test_dir / 'behavior_session_table.csv')
        cls.session_type_map = (
            behavior_sessions_table
            .set_index('behavior_session_id')[['session_type']]
            .to_dict()['session_type'])

        cls.lims_cache = VisualBehaviorOphysProjectCache.from_lims(
            data_release_date=['2021-03-25', '2021-08-12'])
        cls.tempdir = tempdir

        with patch.object(
                BehaviorProjectCloudApi,
                '_get_metadata_path',
                wraps=cls._get_project_table_path):
            cls.cloud_cache = VisualBehaviorOphysProjectCache.from_s3_cache(
                cache_dir=tempdir.name)

    def teardown_class(self):
        self.tempdir.cleanup()

    def _get_behavior_session(self, behavior_session_id, lims_db):
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

    @classmethod
    def _get_project_table_path(cls, fname):
        if fname == 'behavior_session_table':
            return cls.test_dir / 'behavior_session_table.csv'
        elif fname == 'ophys_session_table':
            return cls.test_dir / 'ophys_session_table.csv'
        elif fname == 'ophys_experiment_table':
            return cls.test_dir / 'ophys_experiment_table.csv'
        elif fname == 'ophys_cells_table':
            return cls.test_dir / 'ophys_cells_table.csv'

    @pytest.mark.requires_bamboo
    @pytest.mark.skip('Skipping until data on s3 has been updated')
    def test_behavior_session_table(self):
        with patch.object(
                BehaviorMetadata,
                'from_lims', wraps=self._get_behavior_session):
            from_lims = self.lims_cache.get_behavior_session_table()

        from_lims = from_lims.drop(columns=list(SESSION_SUPPRESS))

        from_s3 = self.cloud_cache.get_behavior_session_table()
        from_s3 = from_s3.drop(columns=['file_id', 'isilon_filepath'])

        from_lims = from_lims.sort_index()
        from_s3 = from_s3.sort_index()

        pd.testing.assert_frame_equal(from_lims, from_s3)

    @pytest.mark.requires_bamboo
    @pytest.mark.skip('Skipping until data on s3 has been updated')
    def test_ophys_session_table(self):
        with patch.object(
                BehaviorMetadata,
                'from_lims', wraps=self._get_behavior_session):
            from_lims = self.lims_cache.get_ophys_session_table()

        from_lims = from_lims.drop(columns=list(SESSION_SUPPRESS))

        from_s3 = self.cloud_cache.get_ophys_session_table()

        from_lims = from_lims.sort_index()
        from_s3 = from_s3.sort_index()

        pd.testing.assert_frame_equal(from_lims, from_s3)

    @pytest.mark.requires_bamboo
    @pytest.mark.skip('Skipping until data on s3 has been updated')
    def test_ophys_experiments_table(self):
        with patch.object(
                BehaviorMetadata,
                'from_lims', wraps=self._get_behavior_session):
            from_lims = self.lims_cache.get_ophys_experiment_table()

        from_lims = from_lims.drop(
            columns=list(OPHYS_EXPERIMENTS_SUPPRESS) +
            list(OPHYS_EXPERIMENTS_SUPPRESS_FINAL), errors='ignore')

        from_s3 = self.cloud_cache.get_ophys_experiment_table()
        from_s3 = from_s3.drop(columns=['file_id', 'isilon_filepath'])

        from_lims = from_lims.sort_index()
        from_s3 = from_s3.sort_index()
        pd.testing.assert_frame_equal(from_lims, from_s3)
