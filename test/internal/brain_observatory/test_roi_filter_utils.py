import pytest

from allensdk.internal.brain_observatory.roi_filter_utils import (
        get_indices_by_distance)


@pytest.mark.parametrize(
        "tree_points, query_points, expected, exception",
        [
            (
                [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]],
                [[0, 0], [2, 2]],
                [0, 4],
                None
            ),
            (
                [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]],
                [[0, 0.4], [0.1, 0.6]],
                [0, 1],
                pytest.raises(AssertionError,
                              match="Max match distance greater than 0")
            ),
            (
                [],
                [],
                [],
                pytest.raises(ValueError,
                              match=("number of dimensions is incorrect. "
                                     "Expected 2 got 1"))
            )
            ])
def test_get_indices_by_distance(tree_points, query_points,
                                 expected, exception):
    """tests exceptions with simple 2D vectors. Actual code has 5D vectors
    for a basic cell-matching to [minx, miny, maxx, maxy, area]
    """
    if exception is None:
        indices = get_indices_by_distance(query_points, tree_points)
        assert all([e == i for e, i in zip(expected, indices)])
    else:
        with exception:
            indices = get_indices_by_distance(query_points, tree_points)
            assert all([e == i for e, i in zip(expected, indices)])
