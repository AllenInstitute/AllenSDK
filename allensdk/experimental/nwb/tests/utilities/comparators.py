
import numpy as np
from pynwb import NWBContainer, NWBData


DICT_LIKE_TYPES = (dict,)
LIST_LIKE_TYPES = (list, tuple)
NDARRAY_LIKE_TYPES = (np.ndarray,)


def nwb_container_assert_equal(left, right):

    left_keys = list(left.__nwbfields__)
    right_keys = list(right.__nwbfields__)

    assert set(left_keys) == set(right_keys)
    for key in left_keys:
        generic_assert_equal(getattr(left, key), getattr(right, key))


def nwb_data_assert_equal(left, right):
    list_like_assert_equal(left, right)


def dict_like_assert_equal(left, right):
    ''' Compares dict-like objects. Each must expose:

        keys()
        __getitem__
    '''
    
    left_keys = list(left.keys())
    right_keys = list(right.keys())

    assert set(left_keys) == set(right_keys)
    for key in left_keys:
        generic_equal(left[key], right[key])


def list_like_assert_equal(left, right):
    for left_item, right_item in zip(left, right):
        generic_assert_equal(left_item, right_item)


def array_like_assert_equal(left, right):
    assert np.array_equal(left, right)


def direct_assert_equal(left, right):
    assert left == right


def generic_assert_equal(left, right):

    if isinstance(left, NWBContainer) and isinstance(right, NWBContainer):
        nwb_container_assert_equal(left, right)
    elif isinstance(left, NWBData) and isinstance(right, NWBData):
        nwb_data_assert_equal(left, right)
    elif isinstance(left, DICT_LIKE_TYPES) and isinstance(right, DICT_LIKE_TYPES):
        dict_assert_equal(left, right)
    elif isinstance(left, LIST_LIKE_TYPES) and isinstance(right, LIST_LIKE_TYPES):
        list_like_assert_equal(left, right)
    elif isinstance(left, NDARRAY_LIKE_TYPES) and isinstance(right, NDARRAY_LIKE_TYPES):
        array_like_assert_equal(left, right)
    elif hasattr(left, '__eq__') and hasattr(right, '__eq__'):
        direct_assert_equal(left, right)
    else:
        raise TypeError('unable to compare types: {}, {}'.format(type(left), type(right)))

