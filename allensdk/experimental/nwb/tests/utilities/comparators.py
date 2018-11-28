from typing import Tuple, Union, Any, Dict, Set
import sys

import pynwb
import numpy as np

from .comparison_types import ComparatorType, integer_types, scalar_numeric_types, list_like_types, catchall_types


def basic_compare_types(
    left_type,  # type: type
    right_type,  # type: type
    subclasses_pass=False  # type: bool
):
    # type: (...) -> Tuple[bool, str]

    if not subclasses_pass:
        result = left_type == right_type
    else:
        result = issubclass(left_type, right_type) or issubclass(right_type, left_type)

    diff = 'left type: {}, right type: {}'.format(left_type, right_type)
    return result, diff


def dict_like_comparator(
    left,  # type: Dict[Any, Any]
    right  # type: Dict[Any, Any]
):
    # type: (...) -> Tuple[bool, str]

    left_keys = set(left.keys())  # type: Set[Any]
    right_keys = set(right.keys())  # type: Set[Any]

    left_extra = left_keys - right_keys
    right_extra = right_keys - left_keys

    if left_keys == right_keys:
        return True, ''
    return False, 'found {} extra keys in the left item: {}\nand {} extra keys in the right item'.format(
        len(left_extra), left_extra, len(right_extra), right_extra
    )


def list_like_comparator(
    left,
    right
):
    # type: (...) -> Tuple[bool, str]
    return len(left) == len(right), '{} elements in left, {} in right'.format(len(left), len(right))


def ndarray_like_comparator(
    left,
    right
):
    # type: (...) -> Tuple[bool, str]
    return np.array_equal(left, right), ''


def nwbdata_like_comparator(
    left,
    right
):
    # type: (...) -> Tuple[bool, str]
    left_arr = np.array(left[:])
    right_arr = np.array(right[:])

    if left_arr.dtype != np.dtype('O') and right_arr.dtype != np.dtype('O'):
        result = np.allclose(left_arr, right_arr)
    elif isinstance(left_arr[0], pynwb.NWBContainer) and isinstance(right_arr[0], pynwb.NWBContainer):
        # this shows up if we've got links to other containers in the file, so we just compare names
        result = all([nwb_container_comparator(l, r) for l, r in zip(left_arr, right_arr)]) 
    else:
        result = np.array_equal(left_arr, right_arr)
    return result, 'left name: {}, right name: {}'.format(left.name, right.name)


def vector_index_like_comparator(
    left,
    right
):

    comparisons = [np.array_equal(l, r) for l, r in zip(left[0::1], right[0::1])]
    data_equal = all(comparisons)
    return data_equal, 'left name: {}, right name: {}'.format(left.name, right.name)


def nwb_container_comparator(
    left,  # type: pynwb.NWBContainer
    right  # type: pynwb.NWBContainer
):
    # type: (...) -> Tuple[bool, str]
    return left.name == right.name, 'left name: {}, right name: {}'.format(left.name, right.name)



def int_like_scalar_comparator(
    left,
    right
):
    # type: (...) -> Tuple[bool, str]
    return left == right, '{} vs {}'.format(left, right)


def float_like_scalar_comparator(
    left,
    right
):
    # type: (...) -> Tuple[bool, str]
    return abs(left - right) <= sys.float_info.epsilon, '{} vs {}'.format(left, right)


def catchall_direct_comparator(
    left,
    right
):
    return left == right, '{} vs {}'.format(left, right)


def no_matching_comparator(
    left,  # type: Any
    right  # type: Any
):
    # type: (...) -> Tuple[bool, str]
    return False, 'no comparator defined for {} and {}'.format(type(left), type(right))


def default_comparator_lookup(
    left_type,  # type: type
    right_type  # type: type
):
    # type: (...) -> ComparatorType

    if issubclass(left_type, dict) and issubclass(right_type, dict):
        return dict_like_comparator
    elif issubclass(left_type, integer_types) and issubclass(right_type, integer_types):
        return int_like_scalar_comparator
    elif issubclass(left_type, scalar_numeric_types) and issubclass(right_type, scalar_numeric_types):
        return float_like_scalar_comparator
    elif issubclass(left_type, list_like_types) and issubclass(right_type, list_like_types):
        return list_like_comparator
    elif issubclass(left_type, np.ndarray) and issubclass(right_type, np.ndarray):
        return ndarray_like_comparator
    elif issubclass(left_type, pynwb.core.VectorIndex) and issubclass(right_type, pynwb.core.VectorIndex):
        return vector_index_like_comparator
    elif issubclass(left_type, pynwb.NWBData) and issubclass(right_type, pynwb.NWBData):
        return nwbdata_like_comparator
    elif issubclass(left_type, pynwb.NWBContainer) and issubclass(right_type, pynwb.NWBContainer):
        return nwb_container_comparator
    elif issubclass(left_type, catchall_types) and issubclass(right_type, catchall_types) and left_type == right_type: # insert new cases above
        return catchall_direct_comparator

    return no_matching_comparator
