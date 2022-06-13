# This file implements tests of the sync-manipulation
# utilities imported from ecephys_etl_pipelines. The purpose
# of these tests is to detect if the behavior of the utilities
# in ecephys_etl_pipelines ever changes out from under us

import pytest
import numpy as np

from allensdk.brain_observatory.sync_dataset import Dataset as SyncDataset
from allensdk.brain_observatory.sync_stim_aligner import (
    _choose_line,
    _get_rising_times,
    _get_falling_times,
    _get_line_starts_and_ends)


def test_choose_line(
        sync_file_fixture):
    """
    Test that _choose_line chooses the expected line
    """
    with SyncDataset(sync_file_fixture) as data:
        assert _choose_line(data, 'lineC') == 'lineC'
        assert _choose_line(data, ('lineB', 'lineA')) == 'lineB'
        assert _choose_line(data, ('lineA', 'lineB')) == 'lineA'
        assert _choose_line(data, ('xxxx', 'lineD')) == 'lineD'
        with pytest.raises(RuntimeError, match='Could not find one of'):
            _choose_line(data, ('xxxx', 'yyyy'))
        with pytest.raises(RuntimeError, match='Could not find one of'):
            _choose_line(data, 'zzzz')


@pytest.mark.parametrize(
        "specified_lines, expected_line",
        [('lineD', 'lineD'),
         (('nonsense', 'lineC'), 'lineC'),
         (('lineC', 'lineB'), 'lineC'),
         (('lineB', 'lineC'), 'lineB')])
def test_get_rising_times(
        sync_file_fixture,
        sync_freq_fixture,
        sync_sample_fixture,
        line_to_edges_fixture,
        specified_lines,
        expected_line):
    """
    Test that _get_rising_times returns the expected timestamp arrays
    """

    with SyncDataset(sync_file_fixture) as data:
        actual = _get_rising_times(
                        data=data,
                        sync_lines=specified_lines)

        expected_idx = line_to_edges_fixture[expected_line]['rising_idx'][1:]
        expected_time = sync_sample_fixture[expected_idx]/sync_freq_fixture
        np.testing.assert_allclose(expected_time, actual)


@pytest.mark.parametrize(
        "specified_lines",
        ['lineZ', ('lineU', 'lineW')])
def test_get_rising_times_exception(
        sync_file_fixture,
        specified_lines):
    """
    Test that _get_rising_times raises the expected
    exception when you specify non-existent lines
    """

    with SyncDataset(sync_file_fixture) as data:
        with pytest.raises(RuntimeError, match="Could not find one of"):
            _get_rising_times(
                        data=data,
                        sync_lines=specified_lines)


@pytest.mark.parametrize(
        "specified_lines, expected_line",
        [('lineD', 'lineD'),
         (('nonsense', 'lineC'), 'lineC'),
         (('lineC', 'lineB'), 'lineC'),
         (('lineB', 'lineC'), 'lineB')])
def test_get_falling_times(
        sync_file_fixture,
        sync_freq_fixture,
        sync_sample_fixture,
        line_to_edges_fixture,
        specified_lines,
        expected_line):
    """
    Test that _get_falling_times returns the expected timestamp arrays
    """

    with SyncDataset(sync_file_fixture) as data:
        actual = _get_falling_times(
                        data=data,
                        sync_lines=specified_lines)

        expected_idx = line_to_edges_fixture[expected_line]['falling_idx'][1:]
        expected_time = sync_sample_fixture[expected_idx]/sync_freq_fixture
        np.testing.assert_allclose(expected_time, actual)


@pytest.mark.parametrize(
        "specified_lines, expected_line",
        [('lineD', 'lineD'),
         (('nonsense', 'lineC'), 'lineC'),
         (('lineC', 'lineB'), 'lineC'),
         (('lineB', 'lineC'), 'lineB')])
def test_get_line_starts_and_ends(
        sync_file_fixture,
        sync_freq_fixture,
        sync_sample_fixture,
        line_to_edges_fixture,
        specified_lines,
        expected_line):
    """
    Test that _get_line_starts_and_ends works as expected
    """
    with SyncDataset(sync_file_fixture) as data:
        actual = _get_line_starts_and_ends(
                        data=data,
                        sync_lines=specified_lines)
        start_idx = line_to_edges_fixture[expected_line]['rising_idx'][1:]
        end_idx = line_to_edges_fixture[expected_line]['falling_idx'][1:]
        start_times = sync_sample_fixture[start_idx]/sync_freq_fixture
        end_times = sync_sample_fixture[end_idx]/sync_freq_fixture
        np.testing.assert_allclose(actual[0], start_times)
        np.testing.assert_allclose(actual[1], end_times)


@pytest.mark.parametrize(
        "specified_lines",
        ['lineZ', ('lineU', 'lineW')])
def test_get_falling_times_exception(
        sync_file_fixture,
        specified_lines):
    """
    Test that _get_falling_times raises the expected
    exception when you specify non-existent lines
    """

    with SyncDataset(sync_file_fixture) as data:
        with pytest.raises(RuntimeError, match="Could not find one of"):
            _get_falling_times(
                        data=data,
                        sync_lines=specified_lines)
