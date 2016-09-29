# Copyright 2016 Allen Institute for Brain Science
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
from allensdk.brain_observatory.session_analysis import SessionAnalysis
import os


@pytest.fixture
def session_a():
    filename = '/data/informatics/keithg/out_510390912.nwb'
    save_path = 'xyza'
    
    sa = SessionAnalysis(filename, save_path)
    
    return sa


@pytest.fixture
def session_b():
    filename = '/data/informatics/keithg/506278598.nwb'
    save_path = 'xyzb'
    
    sa = SessionAnalysis(filename, save_path)
    
    return sa


@pytest.fixture
def session_c():
    filename = '/data/informatics/keithg/out_510221121.nwb'
    save_path = 'xyzc'
    
    sa = SessionAnalysis(filename, save_path)
    
    return sa


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_session_a(session_a):
    session_a.session_a()
      
    assert True


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_session_b(session_b):
    session_b.session_b()
      
    assert True


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial_testing")
def test_session_c(session_c):
    session_c.session_c()
      
    assert True


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_session_get_session_type_a(session_a):
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
