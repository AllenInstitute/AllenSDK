import pytest
import numpy as np
from allensdk.brain_observatory.ecephys.utils import (
    strip_substructure_acronym)


def test_strip_substructure_acronym():
    """
    Test that strip_substructure_acronym behaves properly
    """

    assert strip_substructure_acronym('abcde-fg-hi') == 'abcde'
    assert strip_substructure_acronym(None) is None

    data = ['DG-mo', 'DG-pd', 'LS-ab', 'LT-x', 'AB-cd',
            'WX-yz', 'AB-ef']
    expected = ['AB', 'DG', 'LS', 'LT', 'WX']
    assert strip_substructure_acronym(data) == expected

    data = [None, 'DG-mo', 'DG-pd', 'LS-ab', 'LT-x', 'AB-cd',
            'WX-yz', None, 'AB-ef', np.NaN]
    expected = ['AB', 'DG', 'LS', 'LT', 'WX']
    assert strip_substructure_acronym(data) == expected

    assert strip_substructure_acronym([None]) == []

    assert strip_substructure_acronym(np.NaN) is None

    # pass in a tuple; check that it fails since that is not
    # a str or a list
    with pytest.raises(RuntimeError, match="list or a str"):
        strip_substructure_acronym(('a', 'b', 'c'))

    with pytest.raises(RuntimeError, match="list or a str"):
        strip_substructure_acronym(['abc', 2.3])
