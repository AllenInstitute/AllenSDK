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
from allensdk.api.queries.ontologies_api import OntologiesApi
import pandas as pd
from numpy import allclose
import pytest
from mock import patch


@pytest.fixture
def ontologies():
    return OntologiesApi()


@patch.object(OntologiesApi, "json_msg_query")
def test_get_structure_graph(mock_json_msg_query, ontologies):
    structure_graph_id = 1
    ontologies.get_structures(structure_graph_id)
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,[graph_id$in1],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_list_structure_graphs(mock_json_msg_query, ontologies):
    ontologies.get_structure_graphs()
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureGraph,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_list_structure_sets_noarg(mock_json_msg_query, ontologies):
    ontologies.get_structure_sets()
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureSet,rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_list_structure_sets_args(mock_json_msg_query, ontologies):
    ontologies.get_structure_sets([2, 3])
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::StructureSet,rma::criteria,[id$in2,3],"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_list_atlases(mock_json_msg_query, ontologies):
    ontologies.get_atlases()
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_structure_graph_by_name(mock_json_msg_query, ontologies):
    ontologies.get_structures(structure_graph_names="'Mouse Brain Atlas'")
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "graph[structure_graphs.name$in'Mouse Brain Atlas'],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_structure_graphs_by_names(mock_json_msg_query, ontologies):
    ontologies.get_structures(structure_graph_names=["'Mouse Brain Atlas'",
                                                     "'Human Brain Atlas'"])
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "graph[structure_graphs.name$in'Mouse Brain Atlas',"
        "'Human Brain Atlas'],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_structure_set_by_id(mock_json_msg_query, ontologies):
    ontologies.get_structures(structure_set_ids=8)
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,[structure_set_id$in8],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_structure_sets_by_ids(mock_json_msg_query, ontologies):
    ontologies.get_structures(structure_set_ids=[7, 8])
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,[structure_set_id$in7,8],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_structure_set_by_name(mock_json_msg_query, ontologies):
    ontologies.get_structures(
        structure_set_names=ontologies.quote_string(
            "Mouse Connectivity - Summary"))
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "structure_sets[name$in'Mouse Connectivity - Summary'],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_structure_set_by_names(mock_json_msg_query, ontologies):
    ontologies.get_structures(
        structure_set_names=[
            ontologies.quote_string("NHP - Coarse"),
            ontologies.quote_string("Mouse Connectivity - Summary")])
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "structure_sets[name$in'NHP - Coarse','Mouse Connectivity - Summary'],"
        "rma::options"
        "[num_rows$eq'all'][order$eqstructures.graph_order][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_structure_set_no_order(mock_json_msg_query, ontologies):
    ontologies.get_structures(1, order=None)
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Structure,rma::criteria,"
        "[graph_id$in1],rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_atlas_1(mock_json_msg_query, ontologies):
    atlas_id = 1
    ontologies.get_atlases_table(atlas_id)
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::criteria,"
        "[id$in1],structure_graph(ontology),graphic_group_labels,"
        "rma::include,structure_graph(ontology),graphic_group_labels,"
        "rma::options[only$eq'atlases.id,atlases.name,atlases.image_type,"
        "ontologies.id,ontologies.name,"
        "structure_graphs.id,structure_graphs.name,"
        "graphic_group_labels.id,graphic_group_labels.name']"
        "[num_rows$eq'all'][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_atlas_verbose(mock_json_msg_query, ontologies):
    ontologies.get_atlases_table(brief=False)
    mock_json_msg_query.assert_called_once_with(
        "http://api.brain-map.org/api/v2/data/query.json?q="
        "model::Atlas,rma::criteria,"
        "structure_graph(ontology),graphic_group_labels,"
        "rma::include,structure_graph(ontology),graphic_group_labels,"
        "rma::options[num_rows$eq'all'][count$eqfalse]")


@patch.object(OntologiesApi, "json_msg_query")
def test_get_structures_with_sets(mock_json_msg_query, ontologies):
    ontologies.get_structures_with_sets(1)
    mock_json_msg_query.assert_called_once_with(
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
