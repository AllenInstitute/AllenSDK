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
import os
import numpy as np
from mock import call, patch, mock_open, MagicMock
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi
import json
import allensdk.brain_observatory.stimulus_info as si
from allensdk.test_utilities.regression_fixture import get_list_of_path_dict

try:
    import __builtin__ as builtins  # @UnresolvedImport
except:
    import builtins  # @UnresolvedImport


CACHE_MANIFEST = """
{
  "manifest": [
    {
      "type": "manifest_version",
      "value": "1.3"
    },
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
    },
    {
      "parent_key": "BASEDIR",
      "type": "file",
      "spec": "ophys_analysis_data/%d_%s_analysis.h5",
      "key": "ANALYSIS_DATA"
    },
    {
      "parent_key": "BASEDIR",
      "type": "file",
      "spec": "ophys_experiment_events/%d_events.npz",
      "key": "EVENTS_DATA"
    },
    {
      "parent_key": "BASEDIR",
      "type": "file",
      "spec": "ophys_eye_gaze_mapping/%d_eyetracking_dlc_to_screen_mapping.h5",
      "key": "EYE_GAZE_DATA"
    }
  ]
}
"""


@pytest.fixture()
def events_test_data():
    return {"pattern": "/allen/aibs/informatics/module_test_data/observatory/events/%d_events.npz",
            "experiment_id": 715923832}


@pytest.fixture(scope="function")
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
            boc = BrainObservatoryCache(manifest_file="some_path/manifest.json",
                                        base_uri='http://api.brain-map.org')

    return boc


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_all_targeted_structures(mock_json_msg_query,
                                     brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                brain_observatory_cache.get_all_targeted_structures()

        mock_json_msg_query.assert_called_once_with(
            "http://api.brain-map.org/api/v2/data/query.json?q="
            "model::ExperimentContainer,rma::include,"
            "ophys_experiments,isi_experiment,"
            "specimen(donor(conditions,age,transgenic_lines)),"
            "targeted_structure,"
            "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_experiment_containers(mock_json_msg_query,
                                   brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download experiment containers for VISp experiments
                visp_ecs = brain_observatory_cache.get_experiment_containers(
                    targeted_structures=['VISp'])

    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::ExperimentContainer,rma::include,"
        "ophys_experiments,isi_experiment,"
        "specimen(donor(conditions,age,transgenic_lines)),targeted_structure,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_all_cre_lines(mock_json_msg_query,
                           brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download a list of all cre lines
                tls = brain_observatory_cache.get_all_cre_lines()

    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::ExperimentContainer,rma::include,"
        "ophys_experiments,isi_experiment,"
        "specimen(donor(conditions,age,transgenic_lines)),targeted_structure,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_ophys_experiments(mock_json_msg_query,
                               brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download a list of all transgenic driver lines
                tls = brain_observatory_cache.get_ophys_experiments()

    calls = [call("http://api.brain-map.org/api/v2/data/query.json?q="
                  "model::OphysExperiment,rma::include,experiment_container,"
                  "well_known_files(well_known_file_type),targeted_structure,"
                  "specimen(donor(age,transgenic_lines)),"
                  "rma::options[num_rows$eq'all'][count$eqfalse]"),

             call("http://api.brain-map.org/api/v2/data/query.json?q="
                  "model::WellKnownFile,rma::criteria,well_known_file_type[name$eqEyeDlcScreenMapping],"
                  "rma::options[num_rows$eq'all'][count$eqfalse]")]

    mock_json_msg_query.assert_has_calls(calls)


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_all_session_types(mock_json_msg_query,
                               brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download a list of all transgenic driver lines
                tls = brain_observatory_cache.get_all_session_types()

    calls = [call("http://api.brain-map.org/api/v2/data/query.json?q="
                  "model::OphysExperiment,rma::include,experiment_container,"
                  "well_known_files(well_known_file_type),targeted_structure,"
                  "specimen(donor(age,transgenic_lines)),"
                  "rma::options[num_rows$eq'all'][count$eqfalse]"),

             call("http://api.brain-map.org/api/v2/data/query.json?q="
                  "model::WellKnownFile,rma::criteria,well_known_file_type[name$eqEyeDlcScreenMapping],"
                  "rma::options[num_rows$eq'all'][count$eqfalse]")]

    mock_json_msg_query.assert_has_calls(calls)


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_stimulus_mappings(mock_json_msg_query,
                               brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            with patch('allensdk.core.json_utilities.read',
                       MagicMock(name='read_json')):
                # Download a list of all transgenic driver lines
                tls = brain_observatory_cache._get_stimulus_mappings()

    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::ApiCamStimulusMapping,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@pytest.mark.skipif(True, reason="need to develop mocks")
@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_cell_specimens(mock_json_msg_query,
                            brain_observatory_cache):
    with patch('os.path.exists') as m:
        m.return_value = False

        with patch('allensdk.core.json_utilities.write',
                   MagicMock(name='write_json')):
            # Download a list of all transgenic driver lines
            tls = brain_observatory_cache.get_cell_specimens()

    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q=")


# NOTE: This test should be updated when ugly hack for associating
# ophys experiment id with ophys session id is resolved.
@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_ophys_pupil_data(mock_json_msg_query,
                              brain_observatory_cache):

    with patch.dict('allensdk.core.ophys_experiment_session_id_mapping.ophys_experiment_session_id_map', {111: 777}, clear=True):
        # We are only testing that rma query is correct
        try:
            tls = brain_observatory_cache.get_ophys_pupil_data(111, suppress_pupil_data=False)
        except Exception:
            pass

    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::WellKnownFile,"
        "rma::criteria,[attachable_id$eq777],well_known_file_type[name$eqEyeDlcScreenMapping],"
        "rma::options[num_rows$eq'all'][count$eqfalse]"
    )


def test_build_manifest(tmpdir_factory):
    try:
        manifest_data = bytes(CACHE_MANIFEST, 'UTF-8')  # Python 3
    except:
        manifest_data = bytes(CACHE_MANIFEST)  # Python 2.7

    manifest_file = str(tmpdir_factory.mktemp("boc").join("manifest.json"))
    with patch('allensdk.config.manifest_builder.ManifestBuilder.write_json_string') as mock_write_json_string:
        mock_write_json_string.return_value = manifest_data

        brain_observatory_cache = BrainObservatoryCache(manifest_file=manifest_file)
        with open(manifest_file, 'rb') as f:
            read_manifest_data = f.read()

        assert manifest_data == read_manifest_data


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

@pytest.mark.skipif(not os.path.exists('/allen/aibs/informatics/module_test_data'), reason='AIBS path not available')
@pytest.mark.parametrize("path_dict", get_list_of_path_dict())
def test_brain_observatory_cache_get_analysis_file(brain_observatory_cache, path_dict): 

    nwb_path_pattern = os.path.join(os.path.dirname(path_dict['nwb_file']), '%d.nwb') 
    brain_observatory_cache.manifest.add_path(brain_observatory_cache.EXPERIMENT_DATA_KEY, nwb_path_pattern)

    analysis_path_pattern = os.path.join(os.path.dirname(path_dict['analysis_file']), '%d_%s_analysis.h5') 
    brain_observatory_cache.manifest.add_path(brain_observatory_cache.ANALYSIS_DATA_KEY, analysis_path_pattern)

    oeid = path_dict['ophys_experiment_id']
    data_set = brain_observatory_cache.get_ophys_experiment_data(oeid)
    for stimulus in data_set.list_stimuli():
        if stimulus != si.SPONTANEOUS_ACTIVITY:
            brain_observatory_cache.get_ophys_experiment_analysis(oeid, stimulus)


@pytest.mark.skipif(not os.path.exists('/allen/aibs/informatics/module_test_data'), reason='AIBS path not available')
def test_brain_observatory_cache_get_events_data(brain_observatory_cache, events_test_data):
    eid = events_test_data["experiment_id"]
    data_file = events_test_data["pattern"] % eid

    brain_observatory_cache.manifest.add_path(brain_observatory_cache.EVENTS_DATA_KEY, events_test_data["pattern"])

    events = brain_observatory_cache.get_ophys_experiment_events(eid)
    true_events = np.load(data_file, allow_pickle=False)["ev"]
    assert(np.all(events == true_events))
