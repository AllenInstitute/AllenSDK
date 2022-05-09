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
