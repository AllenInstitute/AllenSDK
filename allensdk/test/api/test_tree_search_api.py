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
####test AllenSDK tree search api for Specimen and Structure
from allensdk.api.queries.tree_search_api import TreeSearchApi
import pytest
import json
from mock import MagicMock

@pytest.fixture
def tree_search():
    tsa = TreeSearchApi()
    tsa.json_msg_query = MagicMock(name='json_msg_query')

    return tsa

def test_get_specimen_tree(tree_search):
    ####ancestor true for Specimen
    kind = 'Specimen'
    db_id = 113817886
    ancestors = True
    descendants = None
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Specimen/113817886.json?ancestors=true")
    
    ####ancestor true for Specimen
    kind = 'Specimen'
    db_id = 113817886
    ancestors = True
    descendants = False
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Specimen/113817886.json?ancestors=true&descendants=false")
    
    ####ancestor false for Specimen
    kind = 'Specimen'
    db_id = 113817886
    ancestors = False
    descendants = True
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Specimen/113817886.json?ancestors=false&descendants=true")

def test_get_structure_tree(tree_search):
    ####ancestor True for Structure
    kind = 'Structure'
    db_id = 12547
    ancestors = True
    descendants = True
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Structure/12547.json?ancestors=true&descendants=true")
    
    ####ancestor False for Structure
    kind = 'Structure'
    db_id = 12547
    ancestors = False
    descendants = True
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Structure/12547.json?ancestors=false&descendants=true")
    
    ####ancestor None for Structure
    kind = 'Structure'
    db_id = 12547
    ancestors = None
    descendants = None
    tree_search.get_tree(kind, db_id, ancestors, descendants)
    tree_search.json_msg_query.assert_called_with("http://api.brain-map.org/api/v2/tree_search/Structure/12547.json")
