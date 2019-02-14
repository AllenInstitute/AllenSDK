import pytest
import copy as cp

from allensdk.core.lazy_property import LazyProperty, LazyPropertyMixin


class CopyApi(object):
    def get_data(self, original_data):
        return cp.copy(original_data)


class DataClass(LazyPropertyMixin):

    def __init__(self, original_data, api=None):
        self.api = CopyApi() if api is None else api
        self.original_data = original_data

        self.data = self.LazyProperty(self.api.get_data, original_data=self.original_data)


@pytest.mark.parametrize('original_data', [{'a': 'b'}, [None]])
def test_first_compute(original_data):
    data_obj = DataClass(original_data)
    assert data_obj.data == original_data
    assert data_obj.data is not original_data
    

@pytest.mark.parametrize('original_data', [1, '1', [None]])
def test_is_lazy(original_data):
    data_obj = DataClass(original_data)

    first = data_obj.data
    second = data_obj.data
    assert first is second


@pytest.mark.parametrize('original_data', [1, '1', [None]])
def test_not_settable(original_data):
    data_obj = DataClass(original_data)
    with pytest.raises(AttributeError) as err:
        data_obj.data = '12345'
        assert "Can't set LazyLoadable attribute" in err