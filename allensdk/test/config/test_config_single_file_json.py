# Copyright 2015-2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.


import pytest
from mock import patch, mock_open
from allensdk.model.biophys_sim.config import Config
try:
    import __builtin__ as builtins  # @UnresolvedImport
except:
    import builtins  # @UnresolvedImport


@pytest.fixture
def simple_config():
    manifest = '''{
        "manifest": [
            { "type": "dir",
              "spec": "MOCK_DOT",
              "key": "BASEDIR"
            }],
        "biophys":
            [{ "hoc": [ "stdgui.hoc"] }]
    }'''

    with patch(builtins.__name__ + ".open",
               mock_open(read_data=manifest)):
        config = Config().load('config.json', False)

    return config


def testAccessHocFilesInData(simple_config):
    assert simple_config.data['biophys'][0]['hoc'][0] == 'stdgui.hoc'


def testManifestIsNotInData(simple_config):
    assert 'manifest' not in simple_config.data


def testManifestInReservedData(simple_config):
    assert 'manifest' in simple_config.reserved_data[0]
