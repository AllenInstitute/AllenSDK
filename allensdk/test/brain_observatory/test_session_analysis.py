# Copyright 2016-2017 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


import pytest
from mock import patch
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
from allensdk.brain_observatory.session_analysis import SessionAnalysis
import os


_orig_get_stimulus_table = BrainObservatoryNwbDataSet.get_stimulus_table


def mock_stimulus_table(dset, name):
    t = _orig_get_stimulus_table(dset, name)
    t.set_value(0, 'end',
                t.loc[0,'start'] + 10)
    
    return t


@pytest.fixture
def session_a():
    filename = '/data/informatics/module_test_data/observatory/test_nwb/out_510390912.nwb'
    save_path = 'xyza'

    sa = SessionAnalysis(filename, save_path)

    return sa


@pytest.fixture
def session_b():
    filename = '/data/informatics/module_test_data/observatory/test_nwb/506278598.nwb'
    save_path = 'xyzb'

    sa = SessionAnalysis(filename, save_path)

    return sa


@pytest.fixture
def session_c():
    filename = '/data/informatics/module_test_data/observatory/test_nwb/out_510221121.nwb'
    save_path = 'xyzc'

    sa = SessionAnalysis(filename, save_path)

    return sa


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
@pytest.mark.parametrize('plot_flag',[False])
def test_session_a(session_a, plot_flag):
    with patch('allensdk.core.brain_observatory_nwb_data_set.BrainObservatoryNwbDataSet.get_stimulus_table',
               mock_stimulus_table):
        session_a.session_a(plot_flag=plot_flag)

        assert True


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
@pytest.mark.parametrize('plot_flag',[False])
def test_session_b(session_b, plot_flag):
    with patch('allensdk.core.brain_observatory_nwb_data_set.BrainObservatoryNwbDataSet.get_stimulus_table',
               mock_stimulus_table):
        session_b.session_b(plot_flag=plot_flag)

        assert True


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial_testing")
@pytest.mark.parametrize('plot_flag',[False])
def test_session_c(session_c, plot_flag):
    with patch('allensdk.core.brain_observatory_nwb_data_set.BrainObservatoryNwbDataSet.get_stimulus_table',
               mock_stimulus_table):
        session_c.session_c(plot_flag=plot_flag)

        assert True


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_session_get_session_type(session_a):
    session_type = session_a.nwb.get_session_type()

    assert session_type == 'three_session_A'


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_session_get_session_type_b(session_b):
    session_type = session_b.nwb.get_session_type()

    assert session_type == 'three_session_B'


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_session_get_session_type_c(session_c):
    session_type = session_c.nwb.get_session_type()

    assert session_type == 'three_session_C'
