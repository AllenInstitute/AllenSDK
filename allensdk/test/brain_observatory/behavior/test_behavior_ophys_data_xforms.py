import pytest

import numpy as np


@pytest.mark.parametrize("roi_ids,expected", [
    [
        1,
        np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])
    ],
    [
        None,
        np.array([
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        ])
    ]
])
# cell_specimen_table_api fixture from allensdk.test.brain_observatory.conftest
def test_get_roi_masks_by_cell_roi_id(roi_ids, expected,
                                      cell_specimen_table_api):
    api = cell_specimen_table_api
    obtained = api.get_roi_masks_by_cell_roi_id(roi_ids)
    assert np.allclose(expected, obtained.values)
    assert np.allclose(obtained.coords['row'],
                       [0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.allclose(obtained.coords['column'],
                       [0.25, 0.75, 1.25, 1.75, 2.25])
