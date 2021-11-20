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
from allensdk.api.queries.glif_api import GlifApi
import numpy as np
import pytest
import os


@pytest.fixture
def neuronal_model_id():
    return 566283950


@pytest.fixture
def specimen_id():
    return 325464516


@pytest.fixture
def glif_api():
    endpoint = None

    if 'TEST_API_ENDPOINT' in os.environ:
        endpoint = os.environ['TEST_API_ENDPOINT']
        return GlifApi(endpoint)
    else:
        return None


@pytest.mark.requires_api_endpoint
@pytest.mark.todo_flaky
def test_get_neuronal_model_templates(glif_api):

    assert len(glif_api.get_neuronal_model_templates()) == 7

    for template in glif_api.get_neuronal_model_templates():

        if template['id'] == 329230710:
            assert 'perisomatic' in template['name']
        elif template['id'] == 395310498:
            assert '(LIF-R-ASC-A)' in template['name']
        elif template['id'] == 395310469:
            assert '(LIF)' in template['name']
        elif template['id'] == 395310475:
            assert '(LIF-ASC)' in template['name']
        elif template['id'] == 395310479:
            assert '(LIF-R)' in template['name']
        elif template['id'] == 471355161:
            assert '(LIF-R-ASC)' in template['name']
        elif template['id'] == 491455321:
            assert 'Biophysical - all active' in template['name']
        else:
            raise Exception('Unrecognized template: %s (%s)' % (template['id'], template['name']))


@pytest.mark.requires_api_endpoint
def test_get_neuronal_models(glif_api, specimen_id):

    cells = glif_api.get_neuronal_models([specimen_id])

    assert len(cells) == 1
    assert len(cells[0]['neuronal_models']) == 2

@pytest.mark.requires_api_endpoint
def test_get_neuronal_models_no_ids(glif_api):
    cells = glif_api.get_neuronal_models()
    assert len(cells) > 0


@pytest.mark.requires_api_endpoint
def test_get_neuron_configs(glif_api, specimen_id):
    model = glif_api.get_neuronal_models([specimen_id])

    neuronal_model_ids = [nm['id'] for nm in model[0]['neuronal_models']]
    assert set(neuronal_model_ids) == set((566283950, 566283946))

    test_id = 566283950 

    np.testing.assert_almost_equal(glif_api.get_neuron_configs([test_id])[test_id]['th_inf'], 0.024561992461740227)

@pytest.mark.requires_api_endpoint
@pytest.mark.todo_flaky
def test_deprecated(fn_temp_dir, glif_api, neuronal_model_id):

    # Exercising deprecated functionality
    len(glif_api.list_neuronal_models())

    glif_api.get_neuronal_model(neuronal_model_id)

    glif_api.get_neuronal_model(neuronal_model_id)
    print(glif_api.get_ephys_sweeps())

    glif_api.get_neuronal_model(neuronal_model_id)
    x = glif_api.get_neuron_config()

    nwb_path = os.path.join(fn_temp_dir, 'tmp.nwb')
    glif_api.get_neuronal_model(neuronal_model_id)
    glif_api.cache_stimulus_file(nwb_path)
