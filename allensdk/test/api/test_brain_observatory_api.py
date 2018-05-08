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
import os
import pytest
from mock import patch, MagicMock, call
from collections import Counter
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi

_rows_per_message = 2000
_msg = [{'whatever': True}] * _rows_per_message
_num_messages = 5
_msg5 = [{'msg': _msg}] * _num_messages


@pytest.fixture()
def bo_api():
    endpoint = os.environ['TEST_API_ENDPOINT'] if 'TEST_API_ENDPOINT' in os.environ else 'http://twarehouse-backup'
    return BrainObservatoryApi(endpoint)


@pytest.fixture
def mock_containers():
    containers = [
        {'targeted_structure': {'acronym': 'CBS'},
         'imaging_depth': 100,
         'specimen': {
            'donor': {'transgenic_lines': [{'name': 'Shiny'}]}}
         },
        {'targeted_structure': {'acronym': 'NBC'},
         'imaging_depth': 200,
         'specimen': {
            'donor': {'transgenic_lines': [{'name': 'Don'}]}}
         }
    ]

    return containers


@pytest.fixture
def mock_ophys_experiments():
    experiments = [
        {'experiment_container_id': 1,
         'targeted_structure': {'acronym': 'CBS'},
         'imaging_depth': 100,
         'specimen': {'donor': {
             'transgenic_lines': [{'name': 'Shiny'}]}},
         'stimulus_name': 'three_session_B',
         },
        {'experiment_container_id': 2,
         'targeted_structure': {'acronym': 'NBC'},
         'imaging_depth': 200,
         'specimen': {'donor': {
             'transgenic_lines': [{'name': 'Don'}]}},
         'stimulus_name': 'three_session_C',
         'experiment_container': { 'failed': False },
         'fail_eye_tracking': False
         },
        {'experiment_container_id': 2,
         'targeted_structure': {'acronym': 'NBC'},
         'imaging_depth': 200,
         'specimen': {'donor': {
             'transgenic_lines': [{'name': 'Don'}]}},
         'stimulus_name': 'three_session_C',
         'experiment_container': { 'failed': True },
         'fail_eye_tracking': True
         }
    ]

    return experiments


@pytest.fixture
def mock_specimens():
    specimens = [
        {"experiment_container_id": 511498500,
         "cell_specimen_id": 517394843
         },
        {"experiment_container_id": 511498742,
         "cell_specimen_id": 517398740,
         "failed_experiment_container": False
         },
        {"experiment_container_id": 511498501,
         "cell_specimen_id": 517394874,
         "failed_experiment_container": True
         }
    ]
    return specimens


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_list_isi_experiments(mock_json_msg_query, bo_api):
    bo_api.list_isi_experiments()
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::IsiExperiment,rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_isi_experiments(mock_json_msg_query, bo_api):
    isi_experiment_id = 503316697
    bo_api.get_isi_experiments(isi_experiment_id)
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::IsiExperiment,rma::criteria,[id$in503316697],"
        "rma::include,"
        "experiment_container(ophys_experiments,targeted_structure),"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_ophys_experiments_one_id(mock_json_msg_query, bo_api):
    ophys_experiment_id = 502066273
    bo_api.get_ophys_experiments(ophys_experiment_id)
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::OphysExperiment,rma::criteria,[id$in502066273],"
        "rma::include,experiment_container,"
        "well_known_files(well_known_file_type),targeted_structure,"
        "specimen(donor(age,transgenic_lines)),"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_experiment_container_metrics(mock_json_msg_query, bo_api):
    tid = 511510627
    bo_api.get_experiment_container_metrics(tid)
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ApiCamExperimentContainerMetric,"
        "rma::criteria,[id$in511510627],"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_experiment_containers(mock_json_msg_query, bo_api):
    tid = 511510753
    bo_api.get_experiment_containers(tid)
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ExperimentContainer,rma::criteria,[id$in511510753],"
        "rma::include,ophys_experiments,isi_experiment,"
        "specimen(donor(conditions,age,transgenic_lines)),targeted_structure,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_column_definitions(mock_json_msg_query, bo_api):
    api_class_name = bo_api.quote_string('ApiTbiDonorMetric')
    bo_api.get_column_definitions(api_class_name=api_class_name)
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ApiColumnDefinition,"
        "rma::criteria,[api_class_name$eq'ApiTbiDonorMetric'],"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_list_column_definition_class_names(mock_json_msg_query, bo_api):
    bo_api.list_column_definition_class_names()
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ApiColumnDefinition,"
        "rma::options"
        "[only$eq'api_class_name'][num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_stimulus_mappings_no_ids(mock_json_msg_query, bo_api):
    bo_api.get_stimulus_mappings()
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ApiCamStimulusMapping,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_stimulus_mappings_one_id(mock_json_msg_query, bo_api):
    ids = 15
    bo_api.get_stimulus_mappings(stimulus_mapping_ids=ids)
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ApiCamStimulusMapping,"
        "rma::criteria,[id$in15],"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_stimulus_mappings_two_ids(mock_json_msg_query, bo_api):
    ids = [15, 43]
    bo_api.get_stimulus_mappings(stimulus_mapping_ids=ids)
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ApiCamStimulusMapping,"
        "rma::criteria,[id$in15,43],"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_cell_metrics_no_ids(mock_json_msg_query, bo_api):
    list(bo_api.get_cell_metrics())
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ApiCamCellMetric,"
        "rma::options[num_rows$eq2000][start_row$eq0][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_cell_metrics_one_ids(mock_json_msg_query, bo_api):
    tid = 517394843
    list(bo_api.get_cell_metrics(cell_specimen_ids=tid))
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ApiCamCellMetric,"
        "rma::criteria,[cell_specimen_id$in517394843],"
        "rma::options[num_rows$eq2000][start_row$eq0][count$eqfalse]")


@patch.object(BrainObservatoryApi, "json_msg_query")
def test_get_cell_metrics_two_ids(mock_json_msg_query, bo_api):
    ids = [517394843, 517394850]
    res = list(bo_api.get_cell_metrics(cell_specimen_ids=ids))
    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::ApiCamCellMetric,"
        "rma::criteria,[cell_specimen_id$in517394843,517394850],"
        "rma::options[num_rows$eq2000][start_row$eq0][count$eqfalse]")


@patch("allensdk.core.json_utilities.read_url_get", side_effect=_msg5)
def test_get_cell_metrics_five_messages(ju_read_url_get, bo_api):
    ids = [517394843, 517394850]
    list(bo_api.get_cell_metrics(cell_specimen_ids=ids))

    base_query = \
       (bo_api.api_url + '/api/v2/data/query.json?q='
        'model::ApiCamCellMetric,'
        'rma::criteria,%5Bcell_specimen_id$in517394843,517394850%5D,'
        'rma::options%5Bnum_rows$eq2000%5D%5Bstart_row$eq{}%5D%5Bcount$eqfalse%5D')
    expected_calls = map(lambda c: call(base_query.format(c)),
                         [0, 2000, 4000, 6000, 8000, 10000])

    assert ju_read_url_get.call_args_list == list(expected_calls)


def test_filter_experiment_containers_no_filters(bo_api, mock_containers):
    containers = bo_api.filter_experiment_containers(mock_containers)
    assert len(containers) == 2


def test_filter_experiment_containers_depth_filter(bo_api, mock_containers):
    containers = bo_api.filter_experiment_containers(mock_containers,
                                                     imaging_depths=[100])
    assert len(containers) == 1


def test_filter_experiment_containers_structures_filter(bo_api, mock_containers):
    containers = \
        bo_api.filter_experiment_containers(
            mock_containers,
            targeted_structures=['CBS'])
    assert len(containers) == 1


def test_filter_experiment_containers_lines_all_filters(bo_api, mock_containers):
    containers = \
        bo_api.filter_experiment_containers(mock_containers,
                                            imaging_depths=[200],
                                            targeted_structures=['NBC'],
                                            transgenic_lines=['Don'])

    assert len(containers) == 1

def test_filter_experiment_containers_caseless(bo_api, mock_containers):
    containers = \
        bo_api.filter_experiment_containers(mock_containers, transgenic_lines=['DON'])
    assert len(containers) == 1



def test_filter_ophys_experiments_no_filters(bo_api, mock_ophys_experiments):
    experiments = bo_api.filter_ophys_experiments(mock_ophys_experiments)
    assert len(experiments) == 2


def test_filter_ophys_experiments_container_id(bo_api, mock_ophys_experiments):
    experiments = bo_api.filter_ophys_experiments(mock_ophys_experiments,
                                                  experiment_container_ids=[1])
    assert len(experiments) == 1


def test_filter_ophys_experiments_stimuli(bo_api, mock_ophys_experiments):
    experiments = bo_api.filter_ophys_experiments(mock_ophys_experiments,
                                                  stimuli=['static_gratings'])
    assert len(experiments) == 1

def test_filter_ophys_experiments_eye_tracking(bo_api, mock_ophys_experiments):
    experiments = bo_api.filter_ophys_experiments(mock_ophys_experiments,
                                                  require_eye_tracking=True)
    assert len(experiments) == 1


def test_filter_cell_specimens(bo_api, mock_specimens):
    specimens = bo_api.filter_cell_specimens(mock_specimens, include_failed=True)
    assert specimens == mock_specimens

    specimens = bo_api.filter_cell_specimens(mock_specimens)
    assert len(specimens) == 2

    specimens = bo_api.filter_cell_specimens(
        mock_specimens, ids=[mock_specimens[0]['cell_specimen_id']])
    assert len(specimens) == 1
    assert specimens[0] == mock_specimens[0]

    cnt = Counter()
    for sp in mock_specimens:
        cnt[sp['experiment_container_id']] += 1

    ecid = mock_specimens[0]['experiment_container_id']
    specimens = bo_api.filter_cell_specimens(
        mock_specimens, experiment_container_ids=[ecid])
    assert len(specimens) == cnt[ecid]
    assert specimens[0] == mock_specimens[0]


@patch.object(BrainObservatoryApi, "retrieve_file_over_http")
@patch.object(BrainObservatoryApi, "json_msg_query", return_value=[{'download_link': '/url/path/to/file'}])
def test_save_ophys_experiment_data(mock_json_msg_query,
                                    mock_retrieve_file_over_http,
                                    bo_api):
    with patch('allensdk.config.manifest.Manifest.safe_mkdir') as mkdir:
        bo_api.save_ophys_experiment_data(1, '/path/to/filename')

        mkdir.assert_called_once_with('/path/to')

    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::WellKnownFile,"
        "rma::criteria,"
        "[attachable_id$eq1],well_known_file_type[name$eqNWBOphys],"
        "rma::options[num_rows$eq'all'][count$eqfalse]")
    mock_retrieve_file_over_http.assert_called_with(
        bo_api.api_url +  '/url/path/to/file',
        '/path/to/filename')


@patch.object(BrainObservatoryApi, "retrieve_file_over_http")
@patch.object(BrainObservatoryApi, "json_msg_query", return_value=[{'download_link': '/url/path/to/file'}])
def test_get_cell_specimen_id_mapping(mock_json_msg_query,
                                      mock_retrieve_file_over_http,
                                      bo_api):
    with patch('pandas.read_csv') as readcsv:
        bo_api.get_cell_specimen_id_mapping('/path/to/filename', 1)

        readcsv.assert_called_once_with('/path/to/filename')

    mock_json_msg_query.assert_called_once_with(
        bo_api.api_url + "/api/v2/data/query.json?q="
        "model::WellKnownFile,"
        "rma::criteria,"
        "[id$eq1],well_known_file_type[name$eqOphysCellSpecimenIdMapping],"
        "rma::options[num_rows$eq'all'][count$eqfalse]")
    mock_retrieve_file_over_http.assert_called_with(
        bo_api.api_url + '/url/path/to/file',
        '/path/to/filename')
