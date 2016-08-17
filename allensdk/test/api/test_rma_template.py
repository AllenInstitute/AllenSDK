# Copyright 2016 Allen Institute for Brain Science
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
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import pytest
from mock import MagicMock
import allensdk.core.json_utilities as ju
from allensdk.api.queries.rma_template import RmaTemplate


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


def test_atlases_list(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'atlases_list')

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::options"
        "%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


def test_structure_graphs_list(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'structure_graphs_list')

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureGraph,rma::options"
        "%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


def test_structure_sets_list(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'structure_sets_list')

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureSet,rma::options"
        "%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


def test_structures_by_graph_ids(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'structures_by_graph_ids',
                       graph_ids='1')

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "%5Bgraph_id$in1%5D,rma::options"
        "%5Bnum_rows$eq%27all%27%5D%5Border$eqstructures.graph_order%5D"
        "%5Bcount$eqfalse%5D")


def test_structures_by_two_graph_ids(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'structures_by_graph_ids',
                       graph_ids=[1, 2])

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "%5Bgraph_id$in1,2%5D,"
        "rma::options"
        "%5Bnum_rows$eq%27all%27%5D"
        "%5Border$eqstructures.graph_order%5D%5Bcount$eqfalse%5D")


def test_structures_by_graph_names(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'structures_by_graph_names',
                       graph_names=rma.quote_string('Human+Brain+Atlas'))

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "graph%5Bstructure_graphs.name$in%27Human+Brain+Atlas%27%5D,"
        "rma::options"
        "%5Bnum_rows$eq%27all%27%5D"
        "%5Border$eqstructures.graph_order%5D%5Bcount$eqfalse%5D")


def test_structures_by_set_ids(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'structures_by_graph_ids',
                       graph_ids='1')

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "%5Bgraph_id$in1%5D,rma::options%5Bnum_rows$eq%27all%27%5D"
        "%5Border$eqstructures.graph_order%5D%5Bcount$eqfalse%5D")


def test_atlases_table(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'atlases_table')

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::criteria,"
        "structure_graph%28ontology%29,graphic_group_labels,"
        "rma::include,%5Bstructure_graph%28ontology%29,graphic_group_labels,"
        "rma::options%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


def test_atlases_table_one_graph(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'atlases_table',
                       graph_ids=1)

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::criteria,"
        "%5Bgraph_id$in1%5D,structure_graph%28ontology%29,graphic_group_labels,"
        "rma::include,%5Bstructure_graph%28ontology%29,graphic_group_labels,"
        "rma::options%5Bnum_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")


def test_atlases_table_brief(rma):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'msg': [{'whatever': True}]})

    rma.template_query('ontology_queries',
                       'atlases_table_brief')

    ju.read_url_get.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,"
        "rma::criteria,structure_graph%28ontology%29,graphic_group_labels,"
        "rma::include,structure_graph%28ontology%29,graphic_group_labels,"
        "rma::options%5Bonly$eq%27atlases.id,atlases.name,atlases.image_type,"
        "ontologies.id,ontologies.name,structure_graphs.id,structure_graphs.name,"
        "graphic_group_labels.id,graphic_group_labels.name%27%5D%5B"
        "num_rows$eq%27all%27%5D%5Bcount$eqfalse%5D")
