# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2016-2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
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
    filename = os.path.abspath(os.path.join(
            "/", "allen", "aibs", "informatics", "module_test_data",
            "observatory", "test_nwb", "out_510390912.nwb"
    ))
    save_path = 'xyza'

    sa = SessionAnalysis(filename, save_path)

    return sa


@pytest.fixture
def session_b():
    filename = os.path.abspath(os.path.join(
            "/", "allen", "aibs", "informatics", "module_test_data", 
            "observatory", "test_nwb", "506278598.nwb"
    ))
    save_path = 'xyzb'

    sa = SessionAnalysis(filename, save_path)

    return sa


@pytest.fixture
def session_c():
    filename = os.path.abspath(os.path.join(
            "/", "allen", "aibs", "informatics", "module_test_data",
            "observatory", "test_nwb", "out_510221121.nwb"
    ))
    save_path = 'xyzc'

    sa = SessionAnalysis(filename, save_path)

    return sa


@pytest.mark.nightly
@pytest.mark.parametrize('plot_flag',[False])
def test_session_a(session_a, plot_flag):
    with patch('allensdk.core.brain_observatory_nwb_data_set.BrainObservatoryNwbDataSet.get_stimulus_table',
               mock_stimulus_table):
        session_a.session_a(plot_flag=plot_flag)

        assert True


@pytest.mark.nightly
@pytest.mark.parametrize('plot_flag',[False])
def test_session_b(session_b, plot_flag):
    with patch('allensdk.core.brain_observatory_nwb_data_set.BrainObservatoryNwbDataSet.get_stimulus_table',
               mock_stimulus_table):
        session_b.session_b(plot_flag=plot_flag)

        assert True


@pytest.mark.nightly
@pytest.mark.parametrize('plot_flag',[False])
def test_session_c(session_c, plot_flag):
    with patch('allensdk.core.brain_observatory_nwb_data_set.BrainObservatoryNwbDataSet.get_stimulus_table',
               mock_stimulus_table):
        session_c.session_c(plot_flag=plot_flag)

        assert True


@pytest.mark.nightly
def test_session_get_session_type(session_a):
    session_type = session_a.nwb.get_session_type()

    assert session_type == 'three_session_A'


@pytest.mark.nightly
def test_session_get_session_type_b(session_b):
    session_type = session_b.nwb.get_session_type()

    assert session_type == 'three_session_B'


@pytest.mark.nightly
def test_session_get_session_type_c(session_c):
    session_type = session_c.nwb.get_session_type()

    assert session_type == 'three_session_C'
