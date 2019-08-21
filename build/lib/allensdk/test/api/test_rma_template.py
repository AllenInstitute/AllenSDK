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
from mock import MagicMock, patch
import allensdk.core.json_utilities as ju
from allensdk.api.queries.rma_template import RmaTemplate


_msg = {'msg': [{'whatever': True}]}


@pytest.fixture
def rma():
    templates = \
        {"ontology_queries": [
            {'name': 'structures_by_graph_ids',
             'description': 'see name',
             'model': 'Structure',
             'criteria': '[graph_id$in{{ graph_ids }}]',
             'order': ['structures.graph_order'],
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['graph_ids']
             },
            {'name': 'structures_by_graph_names',
             'description': 'see name',
             'model': 'Structure',
             'criteria': 'graph[structure_graphs.name$in{{ graph_names }}]',
             'order': ['structures.graph_order'],
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['graph_names']
             },
            {'name': 'structures_by_set_ids',
             'description': 'see name',
             'model': 'Structure',
             'criteria': '[structure_set_id$in{{ set_ids }}]',
             'order': ['structures.graph_order'],
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['set_ids']
             },
            {'name': 'structures_by_set_names',
             'description': 'see name',
             'model': 'Structure',
             'criteria': 'structure_sets[name$in{{ set_names }}]',
             'order': ['structures.graph_order'],
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['set_names']
             },
            {'name': 'structure_graphs_list',
             'description': 'see name',
             'model': 'StructureGraph',
             'num_rows': 'all',
             'count': False
             },
            {'name': 'structure_sets_list',
             'description': 'see name',
             'model': 'StructureSet',
             'num_rows': 'all',
             'count': False
             },
            {'name': 'atlases_list',
             'description': 'see name',
             'model': 'Atlas',
             'num_rows': 'all',
             'count': False
             },
            {'name': 'atlases_table',
             'description': 'see name',
             'model': 'Atlas',
             'criteria': '{% if graph_ids is defined %}[graph_id$in{{ graph_ids }}],{% endif %}structure_graph(ontology),graphic_group_labels',
             'include': '[structure_graph(ontology),graphic_group_labels',
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['graph_ids']
             },
            {'name': 'atlases_table_brief',
             'description': 'see name',
             'model': 'Atlas',
             'criteria': 'structure_graph(ontology),graphic_group_labels',
             'include': 'structure_graph(ontology),graphic_group_labels',
             'only': ['atlases.id',
                      'atlases.name',
                      'atlases.image_type',
                      'ontologies.id',
                      'ontologies.name',
                      'structure_graphs.id',
                      'structure_graphs.name',
                      'graphic_group_labels.id',
                      'graphic_group_labels.name'],
             'num_rows': 'all',
             'count': False
             }
        ]}
    rma = RmaTemplate(query_manifest=templates)

    return rma


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_atlases_list(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'atlases_list')

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::options"
        "%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_structure_graphs_list(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'structure_graphs_list')

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureGraph,rma::options"
        "%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_structure_sets_list(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'structure_sets_list')

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureSet,rma::options"
        "%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_structures_by_graph_ids(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'structures_by_graph_ids',
                       graph_ids='1')

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "%5Bgraph_id$in1%5D,rma::options"
        "%5Bnum_rows$eq%27all%27%5D%5Border$eqstructures.graph_order%5D"
        "%5Bcount$eqfalse%5D")


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_structures_by_two_graph_ids(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'structures_by_graph_ids',
                       graph_ids=[1, 2])

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "%5Bgraph_id$in1,2%5D,"
        "rma::options"
        "%5Bnum_rows$eq%27all%27%5D"
        "%5Border$eqstructures.graph_order%5D%5Bcount$eqfalse%5D")


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_structures_by_graph_names(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'structures_by_graph_names',
                       graph_names=rma.quote_string('Human+Brain+Atlas'))

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "graph%5Bstructure_graphs.name$in%27Human+Brain+Atlas%27%5D,"
        "rma::options"
        "%5Bnum_rows$eq%27all%27%5D"
        "%5Border$eqstructures.graph_order%5D%5Bcount$eqfalse%5D")


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_structures_by_set_ids(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'structures_by_graph_ids',
                       graph_ids='1')

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "%5Bgraph_id$in1%5D,rma::options%5Bnum_rows$eq%27all%27%5D"
        "%5Border$eqstructures.graph_order%5D%5Bcount$eqfalse%5D")


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_atlases_table(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'atlases_table')

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::criteria,"
        "structure_graph%28ontology%29,graphic_group_labels,"
        "rma::include,%5Bstructure_graph%28ontology%29,graphic_group_labels,"
        "rma::options%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_atlases_table_one_graph(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'atlases_table',
                       graph_ids=1)

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::criteria,"
        "%5Bgraph_id$in1%5D,structure_graph%28ontology%29,graphic_group_labels,"
        "rma::include,%5Bstructure_graph%28ontology%29,graphic_group_labels,"
        "rma::options%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


@patch("allensdk.core.json_utilities.read_url_get", return_value=_msg)
def test_atlases_table_brief(ju_read_url_get, rma):
    rma.template_query('ontology_queries',
                       'atlases_table_brief')

    ju_read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,"
        "rma::criteria,structure_graph%28ontology%29,graphic_group_labels,"
        "rma::include,structure_graph%28ontology%29,graphic_group_labels,"
        "rma::options%5Bonly$eq%27atlases.id,atlases.name,atlases.image_type,"
        "ontologies.id,ontologies.name,structure_graphs.id,structure_graphs.name,"
        "graphic_group_labels.id,graphic_group_labels.name%27%5D%5B"
        "num_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")
