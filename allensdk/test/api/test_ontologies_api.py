from allensdk.api.queries.ontologies_api import OntologiesApi

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


def test_list_structure_sets(ontologies):
    ontologies.get_structure_sets()
    
    ontologies.json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureSet,rma::options[num_rows$eq'all'][count$eqfalse]")


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
    ontologies.get_structures(structure_set_ids=[7,8])
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
