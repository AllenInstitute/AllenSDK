import os

import pytest
from mock import MagicMock



def make_biophys_api():
    from allensdk.api.queries.biophysical_api import BiophysicalApi

    endpoint = os.environ['TEST_API_ENDPOINT'] if 'TEST_API_ENDPOINT' in os.environ else 'http://twarehouse-backup'
    return BiophysicalApi(endpoint)


@pytest.fixture
def biophys_api():
    bio = make_biophys_api()
    bio.json_msg_query = MagicMock(name='json_msg_query')

    return bio


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

    print(exp)
    print(obt)


    assert obt == exp
