from typing import Any, Tuple, List, Union, Optional

import pynwb

from .comparison_types import StackerType, StackItemType, list_like_types


def dict_like_stacker(
    left,  # type: Any
    right,  # type: Any
):
    # type: (...) -> Tuple[List[StackItemType], List[StackItemType]]

    left_keys = list(left.keys())

    new_left_stack = []  # type: List[StackItemType]
    new_right_stack = []  # type: List[StackItemType]
    for key in left_keys:
        new_left_stack.append((
            key,
            left[key]
        ))
        new_right_stack.append((
            key,
            right[key]
        ))

    return new_left_stack, new_right_stack


def list_like_stacker(
    left,
    right
):
    # type: (...) -> Tuple[List[StackItemType], List[StackItemType]]

    new_left_stack = []  # type: List[StackItemType]
    new_right_stack = []  # type: List[StackItemType]

    for ii, (left_item, right_item) in enumerate(zip(left, right)):
        new_left_stack.append((
            str(ii),
            left_item
        ))
        new_right_stack.append((
            str(ii),
            right_item
        ))

    return new_left_stack, new_right_stack


def nwb_container_stacker(
    left,  # type: pynwb.NWBContainer
    right  # type: pynwb.NWBContainer
):
    # type: (...) -> Tuple[List[StackItemType], List[StackItemType]]

    new_left_stack = []  # type: List[StackItemType]
    new_right_stack = []  # type: List[StackItemType]

    # handle nwbfields
    for fieldname in left.__nwbfields__:
        new_left_stack.append((
            fieldname,
            getattr(left, fieldname)
        ))
        new_right_stack.append((
            fieldname,
            getattr(right, fieldname)
        ))

    return new_left_stack, new_right_stack


def default_stacker_lookup(
    left_type,  # type: type
    right_type  # type: type 
):
    #  type: (...) -> Optional[StackerType]
    
    if issubclass(left_type, dict) and issubclass(right_type, dict):
        return dict_like_stacker
    if issubclass(left_type, list_like_types) and issubclass(right_type, list_like_types):
        return list_like_stacker
    elif issubclass(left_type, pynwb.NWBContainer) and issubclass(right_type, pynwb.NWBContainer):
        return nwb_container_stacker

    return None