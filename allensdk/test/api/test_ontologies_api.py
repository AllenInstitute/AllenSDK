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

from allensdk.api.queries.ontologies_api import OntologiesApi
import pandas as pd
from numpy import allclose
import pytest
from mock import MagicMock


@pytest.fixture
def ontologies():
    oa = OntologiesApi()
    oa.json_msg_query = MagicMock(name='json_msg_query')

    return oa


def test_get_structure_graph(ontologies):
    structure_graph_id = 1
    ontologies.get_structures(structure_graph_id)
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,[graph_id$in1],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


def test_list_structure_graphs(ontologies):
    ontologies.get_structure_graphs()
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureGraph,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


def test_list_structure_sets_noarg(ontologies):
    ontologies.get_structure_sets()
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureSet,rma::options[num_rows$eq'all'][count$eqfalse]")
        
        
def test_list_structure_sets_args(ontologies):
    ontologies.get_structure_sets([2, 3])
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureSet,rma::criteria,[id$in2,3],"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


def test_list_atlases(ontologies):
    ontologies.get_atlases()
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::options[num_rows$eq'all'][count$eqfalse]")


def test_structure_graph_by_name(ontologies):
    ontologies.get_structures(structure_graph_names="'Mouse Brain Atlas'")
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "graph[structure_graphs.name$in'Mouse Brain Atlas'],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


def test_structure_graphs_by_names(ontologies):
    ontologies.get_structures(structure_graph_names=["'Mouse Brain Atlas'",
                                                     "'Human Brain Atlas'"])
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "graph[structure_graphs.name$in'Mouse Brain Atlas',"
        "'Human Brain Atlas'],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


def test_structure_set_by_id(ontologies):
    ontologies.get_structures(structure_set_ids=8)
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,[structure_set_id$in8],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


def test_structure_sets_by_ids(ontologies):
    ontologies.get_structures(structure_set_ids=[7, 8])
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,[structure_set_id$in7,8],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


def test_structure_set_by_name(ontologies):
    ontologies.get_structures(
        structure_set_names=ontologies.quote_string(
            "Mouse Connectivity - Summary"))
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "structure_sets[name$in'Mouse Connectivity - Summary'],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


def test_structure_set_by_names(ontologies):
    ontologies.get_structures(
        structure_set_names=[
            ontologies.quote_string("NHP - Coarse"),
            ontologies.quote_string("Mouse Connectivity - Summary")])
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "structure_sets[name$in'NHP - Coarse','Mouse Connectivity - Summary'],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


def test_structure_set_no_order(ontologies):
    ontologies.get_structures(1, order=None)
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "[graph_id$in1],rma::options[num_rows$eq'all'][count$eqfalse]")


def test_atlas_1(ontologies):
    atlas_id = 1
    ontologies.get_atlases_table(atlas_id)
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::criteria,"
        "[id$in1],structure_graph(ontology),graphic_group_labels,"
        "rma::include,structure_graph(ontology),graphic_group_labels,"
        "rma::options[only$eq'atlases.id,atlases.name,atlases.image_type,"
        "ontologies.id,ontologies.name,"
        "structure_graphs.id,structure_graphs.name,"
        "graphic_group_labels.id,graphic_group_labels.name']"
        "[num_rows$eq'all'][count$eqfalse]")


def test_atlas_verbose(ontologies):
    ontologies.get_atlases_table(brief=False)
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::criteria,"
        "structure_graph(ontology),graphic_group_labels,"
        "rma::include,structure_graph(ontology),graphic_group_labels,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")
        
        
def test_get_structures_with_sets(ontologies):
    ontologies.get_structures_with_sets(1)
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,[graph_id$in1],"
        "rma::include,structure_sets,"
        "rma::options[num_rows$eq'all'][order$eqstructures.graph_order]"
        "[count$eqfalse]")
        
        
def test_unpack_structure_set_ancestors(ontologies):

    sdf = pd.DataFrame([{'structure_id_path': '/1/2/3/'}])
    ontologies.unpack_structure_set_ancestors(sdf)
    
    assert( 'structure_set_ancestor' in sdf.columns.values )
    assert( allclose(sdf['structure_set_ancestor'].values[0], [1, 2, 3]) )
