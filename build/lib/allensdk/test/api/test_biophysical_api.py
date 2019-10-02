import os
import json

import numpy as np
import pytest
from mock import patch
from allensdk.api.queries.biophysical_api import BiophysicalApi


@pytest.fixture
def neuronal_model_response():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'response_test_data', '472451419_response.json')

    with open(path, 'r') as jf:
        data = json.load(jf)

    return data


@pytest.fixture
def biophys_api():
    endpoint = 'http://twarehouse-backup'
    return BiophysicalApi(endpoint)


@pytest.mark.parametrize('model_id', [3])
@pytest.mark.parametrize('fmt', [None, 'json', 'xml'])
def test_build_rma(model_id, fmt, biophys_api):
    if fmt is None:
        fmt_exp = 'json'
        obt = biophys_api.build_rma(model_id)
    else:
        fmt_exp = fmt
        obt = biophys_api.build_rma(model_id, fmt_exp)

    exp = 'http://twarehouse-backup/api/v2/data/query.{}?'\
          'q=model::NeuronalModel,'\
          'rma::criteria,[id$eq{}],'\
          'neuronal_model_template(well_known_files(well_known_file_type)),'\
          'specimen(ephys_result(well_known_files(well_known_file_type)),'\
          'neuron_reconstructions(well_known_files(well_known_file_type)),ephys_sweeps),'\
          'well_known_files(well_known_file_type),'\
          'rma::include,neuronal_model_template(well_known_files(well_known_file_type)),'\
          'specimen(ephys_result(well_known_files(well_known_file_type)),'\
          'neuron_reconstructions(well_known_files(well_known_file_type)),ephys_sweeps),'\
          'well_known_files(well_known_file_type)'
    exp = exp.format(fmt_exp, model_id)

    assert obt == exp


def test_is_well_known_file_type(biophys_api):
    wkf = {'well_known_file_type': {'name': 'fish'}}

    assert(biophys_api.is_well_known_file_type(wkf, 'fish'))
    assert(not biophys_api.is_well_known_file_type(wkf, 'fowl'))


@patch.object(BiophysicalApi, "json_msg_query")
def test_get_neuronal_models(mock_json_msg_query, biophys_api):
    mck = biophys_api.get_neuronal_models([386049446,469753383])

    mock_json_msg_query.assert_called_once_with(
        "http://twarehouse-backup/api/v2/data/query.json?"
        "q=model::NeuronalModel,rma::criteria,[neuronal_model_template_id$in491455321,329230710],"
        "[specimen_id$in386049446,469753383],rma::options[num_rows$eq'all'][count$eqfalse]")


def test_read_json(biophys_api, neuronal_model_response):
    
    obt = biophys_api.read_json(neuronal_model_response)

    assert(obt['stimulus']['491198851'] == "386049444.nwb")
    assert(obt['morphology']['491459173'] == "Nr5a1-Cre_Ai14-177334.05.01.01_491459171_m.swc")
    assert(obt['fit']['497235805'] == '386049446_fit.json')
    assert(obt['marker']['496607103'] == 'Nr5a1-Cre_Ai14-177334.05.01.01_491459171_marker_m.swc')
    assert(obt['modfiles']['395337293'] == os.path.join('modfiles', 'SK.mod'))
    assert(np.allclose(biophys_api.sweeps, [42]))
