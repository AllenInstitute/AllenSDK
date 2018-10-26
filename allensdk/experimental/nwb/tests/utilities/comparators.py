from typing import Tuple, Union, Any, Dict, Set
import sys

import pynwb

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
    elif issubclass(left_type, pynwb.NWBContainer) and issubclass(right_type, pynwb.NWBContainer):
        return nwb_container_comparator
    elif issubclass(left_type, catchall_types) and issubclass(right_type, catchall_types) and left_type == right_type: # insert new cases above
        return catchall_direct_comparator

    return no_matching_comparator
