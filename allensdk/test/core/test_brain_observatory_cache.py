# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
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
from mock import patch, mock_open, MagicMock
from allensdk.core.brain_observatory_cache import (BrainObservatoryCache, 
                                                   _find_container_tags,
                                                   _merge_transgenic_lines,
                                                   _find_specimen_cre_line,
                                                   _find_specimen_reporter_line)

try:
    import __builtin__ as builtins  # @UnresolvedImport
except:
    import builtins  # @UnresolvedImport


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

    try:
        manifest_data = bytes(CACHE_MANIFEST, 'UTF-8')  # Python 3
    except:
        manifest_data = bytes(CACHE_MANIFEST)  # Python 2.7

    with patch('os.path.exists',
               return_value=True):
        with patch(builtins.__name__ + ".open",
                   mock_open(read_data=manifest_data)):
            # Download a list of all targeted areas
            boc = BrainObservatoryCache(manifest_file='boc/manifest.json',
                                        base_uri='http://api.brain-map.org')

    boc.api.json_msg_query = MagicMock(name='json_msg_query')

    return boc

def test_get_all_targeted_structures(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                brain_observatory_cache.get_all_targeted_structures()

        brain_observatory_cache.api.json_msg_query.assert_called_once_with(
            "http://api.brain-map.org/api/v2/data/query.json?q="
            "model::ExperimentContainer,rma::include,"
            "ophys_experiments,isi_experiment,"
            "specimen(donor(conditions,age,transgenic_lines)),"
            "targeted_structure,"
            "rma::options[num_rows$eq'all'][count$eqfalse]")


def test_get_experiment_containers(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download experiment containers for VISp experiments
                visp_ecs = brain_observatory_cache.get_experiment_containers(
                    targeted_structures=['VISp'])

    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::ExperimentContainer,rma::include,"
        "ophys_experiments,isi_experiment,"
        "specimen(donor(conditions,age,transgenic_lines)),targeted_structure,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


def test_get_all_cre_lines(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download a list of all cre lines
                tls = brain_observatory_cache.get_all_cre_lines()

    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::ExperimentContainer,rma::include,"
        "ophys_experiments,isi_experiment,"
        "specimen(donor(conditions,age,transgenic_lines)),targeted_structure,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


def test_get_ophys_experiments(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download a list of all transgenic driver lines
                tls = brain_observatory_cache.get_ophys_experiments()

    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::OphysExperiment,rma::include,experiment_container,"
        "well_known_files(well_known_file_type),targeted_structure,"
        "specimen(donor(age,transgenic_lines)),"
        "rma::options[num_rows$eq'all'][count$eqfalse]")

def test_get_all_session_types(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download a list of all transgenic driver lines
                tls = brain_observatory_cache.get_all_session_types()

    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::OphysExperiment,rma::include,experiment_container,"
        "well_known_files(well_known_file_type),targeted_structure,"
        "specimen(donor(age,transgenic_lines)),"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


def test_get_stimulus_mappings(brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download a list of all transgenic driver lines
                tls = brain_observatory_cache._get_stimulus_mappings()

    brain_observatory_cache.api.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
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
        "http://api.brain-map.org/api/v2/data/query.json?q=")


def test_build_manifest():
    try:
        manifest_data = bytes(CACHE_MANIFEST, 'UTF-8')  # Python 3
    except:
        manifest_data = bytes(CACHE_MANIFEST)  # Python 2.7

    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.config.manifest.Manifest.safe_mkdir') as mkdir:
            with patch('allensdk.config.manifest_builder.'
                       'ManifestBuilder.write_json_file',
                       MagicMock(name='write_json_file')) as mock_write_json:
                with patch(builtins.__name__ + ".open",
                           mock_open(read_data=manifest_data)):
                    brain_observatory_cache = BrainObservatoryCache(
                        manifest_file='boc/manifest.json',
                        base_uri='http://api.brain-map.org')
                    mkdir.assert_called_once_with('boc')
                    mock_write_json.assert_called_once_with(
                        'boc/manifest.json')


def test_string_argument_errors(brain_observatory_cache):
    boc = brain_observatory_cache

    with pytest.raises(TypeError):
        boc.get_experiment_containers(targeted_structures='str')

    with pytest.raises(TypeError):
        boc.get_experiment_containers(cre_lines='str')

    with pytest.raises(TypeError):
        boc.get_ophys_experiments(targeted_structures='str')

    with pytest.raises(TypeError):
        boc.get_ophys_experiments(cre_lines='str')

    with pytest.raises(TypeError):
        boc.get_ophys_experiments(stimuli='str')

    with pytest.raises(TypeError):
        boc.get_ophys_experiments(session_types='str')

def test_find_container_tags():
    # no conditions no tags
    c = { "specimen": { "donor": { "conditions": [] } } }
    tags = _find_container_tags(c)
    assert len(tags) == 0

    # tissue tags are ignored
    c = { "specimen": { "donor": { "conditions": [ { "name": "tissuecyte" } ] } } }
    tags = _find_container_tags(c)
    assert len(tags) == 0

    # no conditions is okay
    c = { "specimen": { "donor": { } } }
    tags = _find_container_tags(c)
    assert len(tags) == 0

    # everything else goes through
    c = { "specimen": { "donor": { "conditions": [ { "name": "fish" } ] } } }
    tags = _find_container_tags(c)
    assert len(tags) == 1

def test_merge_transgenic_lines():
    # None is allowed and should be ignored
    t1 = [ "a", "b", "c" ]
    t2 = None
    tm = _merge_transgenic_lines(t1,t2)
    assert sorted(tm) == [ "a", "b", "c"]

    # otherwise it's just a merge
    t1 = [ "a", "b", "c" ]
    t2 = [ "c", "d" ]
    tm = _merge_transgenic_lines(t1,t2)
    assert sorted(tm) == [ "a", "b", "c", "d"]

    # one list is fine
    t1 = [ "a", "b", "c" ]
    tm = _merge_transgenic_lines(t1)
    assert sorted(tm) == [ "a", "b", "c" ]

def test_find_specimen_cre_line():
    # None if no TLs
    s = { "donor": { "transgenic_lines": [ ] } }
    cre = _find_specimen_cre_line(s)
    assert cre is None

    # None if no 'Cre'
    s = { "donor": { "transgenic_lines": [ { "transgenic_line_type_name": "driver", "name": "banana" } ] } }
    cre = _find_specimen_cre_line(s)
    assert cre is None

    # None if no 'Cre'
    s = { "donor": { "transgenic_lines": [ { "transgenic_line_type_name": "driver", "name": "bananaCre" } ] } }
    cre = _find_specimen_cre_line(s)
    assert cre == "bananaCre"

    # None if no 'driver'
    s = { "donor": { "transgenic_lines": [ { "transgenic_line_type_name": "reporter", "name": "bananaCre" } ] } }
    cre = _find_specimen_cre_line(s)
    assert cre == None

def test_find_specimen_reporter_line():
    # None if no TLs
    s = { "donor": { "transgenic_lines": [ ] } }
    cre = _find_specimen_reporter_line(s)
    assert cre is None

    s = { "donor": { "transgenic_lines": [ { "transgenic_line_type_name": "reporter", "name": "banana" } ] } }
    cre = _find_specimen_reporter_line(s)
    assert cre == "banana"

    # None if no "reporter"
    s = { "donor": { "transgenic_lines": [ { "transgenic_line_type_name": "driver", "name": "bananaCre" } ] } }
    cre = _find_specimen_reporter_line(s)
    assert cre is None


    
