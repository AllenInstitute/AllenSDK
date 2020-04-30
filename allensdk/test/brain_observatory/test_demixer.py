import numpy as np
import pytest
import scipy.sparse as sparse
import logging

import allensdk.brain_observatory.demixer as dmx


@pytest.mark.parametrize(
    "source_frame,mask_traces,flat_masks,pixels_per_mask,expected",
    [
        (
            np.array([2., 2., 2., 1.]),
            np.array([2.0, 2.0]),
            sparse.csr_matrix(np.array([[1, 0, 0, 0], [1, 1, 0, 0]])),
            np.array([1, 2]),
            np.array([0, 2]),
        ),
        (
            np.array([2., 0., 2., 1.]),
            np.array([2.0, 0.]),    # zero in mask trace
            sparse.csr_matrix(np.array([[1, 0, 0, 0], [1, 1, 0, 0]])),
            np.array([1, 2]),
            None,
        ),
        (
            np.array([2., 0., 2., 1.]),
            np.array([2.0, 0.]),
            sparse.csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 0, 0]])),
            np.array([1, 0]),    # invalid mask (zero pixels)
            None,
        )
    ]
)
def test_demix_point(
        source_frame, mask_traces, flat_masks, pixels_per_mask, expected):
    result = dmx._demix_point(source_frame, mask_traces, flat_masks,
                              pixels_per_mask)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "source_frame,mask_traces,flat_masks,pixels_per_mask,expected",
    [
        (np.zeros(4),   # force singular matrix
         np.ones(2),
         sparse.csr_matrix(np.array([[1, 0, 0, 0], [1, 1, 0, 0]])),
         np.array([1, 2]),
         np.zeros(2)),
    ]
)
def test_demix_raises_warning_for_singular_matrix(
        source_frame, mask_traces, flat_masks, pixels_per_mask, expected,
        caplog):
    result = dmx._demix_point(source_frame, mask_traces, flat_masks,
                              pixels_per_mask)
    with caplog.at_level(logging.WARNING):
        assert caplog.records[0].msg == ("Singular matrix, using least squares to "
                                         "solve.")
        assert caplog.records[0].levelno == logging.WARNING
    np.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    "raw_traces,stack,masks,expected",
    [
        (
            np.array([[2.0, 0.0], [2.0, 2.0]]),
            np.array([[[2., 2.], [2., 1.]], [[2., 2.], [2., 1.]]]),
            np.array([[[1, 0], [0, 0]], [[1, 1], [0, 0]]]),
            (np.array([[0, 0], [2, 0]]), [False, True])
        ),
    ],
)
def test_demix_time_dep_masks(raw_traces, stack, masks, expected):
    result = dmx.demix_time_dep_masks(raw_traces, stack, masks)
    np.testing.assert_equal(result[0], expected[0])
    assert result[1] == expected[1]
