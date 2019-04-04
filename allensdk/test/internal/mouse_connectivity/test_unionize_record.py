

import pytest
import mock

from allensdk.internal.mouse_connectivity.interval_unionize.unionize_record import Unionize


@pytest.mark.parametrize('method', ['__init__', 'calculate', 'propagate', 'output'])
def test_unionize(method):

    un = object.__new__(Unionize)
    
    with pytest.raises(NotImplementedError):
        getattr(un, method)('foo', 'fish')
