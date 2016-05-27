# Copyright 2016 Allen Institute for Brain Science
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
from mock import patch, mock_open, MagicMock
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from _pytest.monkeypatch import monkeypatch
import __builtin__
from allensdk.core import json_utilities

CACHE_MANIFEST = """
{
  "manifest": [
    {
      "type": "dir",
      "spec": ".",
      "key": "BASEDIR"
    },
    {
      "parent_key": "BASEDIR",
      "type": "file",
      "spec": "experiment_containers.json",
      "key": "EXPERIMENT_CONTAINERS"
    },
    {
      "parent_key": "BASEDIR",
      "type": "file",
      "spec": "ophys_experiments.json",
      "key": "EXPERIMENTS"
    },
    {
      "parent_key": "BASEDIR",
      "type": "file",
      "spec": "ophys_experiment_data/%d.nwb",
      "key": "EXPERIMENT_DATA"
    },
    {
      "parent_key": "BASEDIR",
      "type": "file",
      "spec": "cell_specimens.json",
      "key": "CELL_SPECIMENS"
    },
    {
      "parent_key": "BASEDIR",
      "type": "file",
      "spec": "stimulus_mappings.json",
      "key": "STIMULUS_MAPPINGS"
    }
  ]
}
"""


@pytest.fixture
def brain_observatory_cache():
    boc = None
    
    with patch('os.path.exists') as m:
        m.return_value = True
        
        with patch("__builtin__.open",
                   mock_open(read_data=CACHE_MANIFEST)):
            # Download a list of all targeted areas
            boc = BrainObservatoryCache(manifest_file='boc/manifest.json',
                                        base_uri='http://testwarehouse:9000')
            
    boc.api.json_msg_query = MagicMock(name='json_msg_query')
            
    return boc


def test_get_all_targeted_structures(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False
        
        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            targeted_structures = \
                brain_observatory_cache.get_all_targeted_structures()
        
        brain_observatory_cache.api.json_msg_query.assert_called_once_with(
            "http://testwarehouse:9000/api/v2/data/query.json?q="
            "model::ExperimentContainer,rma::include,"
            "ophys_experiments,isi_experiment,"
            "specimen(donor(age,transgenic_lines)),"
            "targeted_structure,"
            "rma::options[num_rows$eq'all'][count$eqfalse]")
    
    
def test_get_experiment_containers(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False
        
        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            # Download experiment containers for VISp experiments
            visp_ecs = brain_observatory_cache.get_experiment_containers(
                targeted_structures=['VISp'])

    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://testwarehouse:9000/api/v2/data/query.json?q="
        "model::ExperimentContainer,rma::include,"
        "ophys_experiments,isi_experiment,"
        "specimen(donor(age,transgenic_lines)),targeted_structure,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


def test_get_all_cre_lines(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False
        
        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):   
            # Download a list of all cre lines 
            tls = brain_observatory_cache.get_all_cre_lines()
            
    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://testwarehouse:9000/api/v2/data/query.json?q="
        "model::ExperimentContainer,rma::include,"
        "ophys_experiments,isi_experiment,"
        "specimen(donor(age,transgenic_lines)),targeted_structure,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")
    
    
def test_get_ophys_experiments(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False
        
        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):   
            # Download a list of all transgenic driver lines 
            tls = brain_observatory_cache.get_ophys_experiments()
            
    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://testwarehouse:9000/api/v2/data/query.json?q="
        "model::OphysExperiment,rma::include,"
        "well_known_files(well_known_file_type),targeted_structure,"
        "specimen(donor(age,transgenic_lines)),"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


def test_get_all_session_types(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False
        
        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):   
            # Download a list of all transgenic driver lines 
            tls = brain_observatory_cache.get_all_session_types()
            
    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://testwarehouse:9000/api/v2/data/query.json?q="
        "model::OphysExperiment,rma::include,"
        "well_known_files(well_known_file_type),targeted_structure,"
        "specimen(donor(age,transgenic_lines)),"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


def test_get_stimulus_mappings(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False
        
        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):   
            # Download a list of all transgenic driver lines 
            tls = brain_observatory_cache._get_stimulus_mappings()
            
    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://testwarehouse:9000/api/v2/data/query.json?q="
        "model::ApiCamStimulusMapping,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@pytest.mark.skipif(True, reason="need to develop mocks")
def test_get_cell_specimens(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False
         
        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            # Download a list of all transgenic driver lines 
            tls = brain_observatory_cache.get_cell_specimens()
            
    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://testwarehouse:9000/api/v2/data/query.json?q=")


def test_build_manifest():
    with patch('os.path.exists') as m:
        m.return_value = False
        
        with patch('allensdk.config.manifest.Manifest.safe_mkdir') as mkdir:
            with patch('allensdk.config.manifest_builder.'
                       'ManifestBuilder.write_json_file',
                       MagicMock(name='write_json_file')) as mock_write_json:
                with patch("__builtin__.open",
                           mock_open(read_data=CACHE_MANIFEST)):
                    brain_observatory_cache = BrainObservatoryCache(
                        manifest_file='boc/manifest.json',
                        base_uri='http://testwarehouse:9000')
                    mkdir.assert_called_once_with('boc')
                    mock_write_json.assert_called_once_with(
                        'boc/manifest.json')
