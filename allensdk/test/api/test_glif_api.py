# Copyright 2016-2017 Allen Institute for Brain Science
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

from allensdk.api.queries.glif_api import GlifApi
import numpy as np
import pytest
import os


@pytest.fixture
def glif_api():
    endpoint = None

    if 'TEST_API_ENDPOINT' in os.environ:
        endpoint = os.environ['TEST_API_ENDPOINT']
        return GlifApi(endpoint)
    else:
        return None


@pytest.mark.skipif(glif_api() is None, reason='No TEST_API_ENDPOINT set.')
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


@pytest.mark.skipif(glif_api() is None, reason='No TEST_API_ENDPOINT set.')
def test_get_neuronal_models(glif_api):

    model = glif_api.get_neuronal_models([325464516])

    assert len(model) == 1
    assert len(model[0]['neuronal_models']) == 2


@pytest.mark.skipif(glif_api() is None, reason='No TEST_API_ENDPOINT set.')
def test_get_neuron_configs(glif_api):
    model = glif_api.get_neuronal_models([325464516])

    neuronal_model_id = model[0]['neuronal_models'][0]['id']
    assert neuronal_model_id == 473465606

    np.testing.assert_almost_equal(glif_api.get_neuron_configs([neuronal_model_id])[neuronal_model_id]['th_inf'], 0.0236421993869)


def test_deprecated():

    # Exercising deprecated functionality
    tmp = GlifApi()
    len(tmp.list_neuronal_models())

    tmp = GlifApi()
    tmp.get_neuronal_model(473465606)

    tmp = GlifApi()
    tmp.get_neuronal_model(473465606)
    print tmp.get_ephys_sweeps()

    tmp = GlifApi()
    tmp.get_neuronal_model(473465606)
    x = tmp.get_neuron_config()

    tmp = GlifApi()
    tmp.get_neuronal_model(473465606)
    tmp.cache_stimulus_file('tmp.nwb')