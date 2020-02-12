import pytest

from allensdk.internal.morphology.morphology import Morphology
from allensdk.internal.morphology.node import Node


def test_apply_affine():
    node_list = [Node(0, 1, 0, 0, 0, 3, -1), Node(1, 2, 0, 0, 1, 1, 0)]
    morph = Morphology(node_list)

    scale = [2, 0, 0,
             0, 2, 0,
             0, 0, 2]
    translate = [1, 0, 0]
    affine = scale + translate
    morph.apply_affine(affine)

    # was at (0, 0, 1) with r = 1
    expected_node1 = {'id': 1,
                      'type': 2,
                      'x': 1,
                      'y': 0,
                      'z': 2,
                      'radius': 2,
                      'parent': 0,
                      'children': [],
                      'tree_id': 0,
                      'compartment_id': 0}

    obtained_node1 = morph.node_list[1]
    for key, value in expected_node1.items():
        assert value == pytest.approx(obtained_node1[key])
