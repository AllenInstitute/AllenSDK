from collections import namedtuple
from typing import Any, Tuple, List

import pandas as pd

from .comparison_types import ComparatorLookupType, StackerLookupType, ComparisonResult, StackItemType, TypeComparatorType
from .comparators import basic_compare_types, default_comparator_lookup
from .stackers import default_stacker_lookup


def compare(
    left,  # type: Any
    right,  # type: Any
    type_comparator=basic_compare_types,  # type: TypeComparatorType
    comparator_lookup=default_comparator_lookup,  # type: ComparatorLookupType
    stacker_lookup=default_stacker_lookup  # type: StackerLookupType
):
    # type: (...) -> Tuple[ComparisonResult, List[StackItemType], List[StackItemType]]

    left_type = type(left)
    right_type = type(right)

    types_equal, types_diff = type_comparator(left_type, right_type)

    direct_comparator = comparator_lookup(left_type, right_type)
    if direct_comparator is not None:
        direct_equal, direct_diff = direct_comparator(left, right)

    stacker = stacker_lookup(left_type, right_type)
    left_stack = []  # type: List[StackItemType]
    right_stack = []  # type: List[StackItemType]
    if stacker is not None:
        left_stack, right_stack = stacker(left, right)

    comparison_result = ComparisonResult(types_equal, direct_equal, types_diff, direct_diff)
    return comparison_result, left_stack, right_stack


def nested_compare(
    left,  # type: Any
    right,  # type: Any
    left_name='initial',  # type: str
    right_name='initial',  # type: str
    type_comparator=basic_compare_types,  # type: TypeComparatorType
    comparator_lookup=default_comparator_lookup,  # type: ComparatorLookupType
    stacker_lookup=default_stacker_lookup  # type: StackerLookupType
):
    
    results = []  # type: List[Tuple[Tuple[str, str], ComparisonResult]]

    left_stack = [(left_name, left)]
    right_stack = [(right_name, right)]

    while len(left_stack) > 0:
        left_name, left = left_stack.pop()
        right_name, right = right_stack.pop()

        comparison_result, new_left, new_right = compare(
            left, 
            right, 
            type_comparator=type_comparator, 
            comparator_lookup=comparator_lookup, 
            stacker_lookup=stacker_lookup
        )

        results.append((
            (left_name, right_name),
            comparison_result
        ))

        left_stack.extend(new_left)
        right_stack.extend(new_right)

    return results


def comparisons_as_df(
    comparison_results  # type: List[Tuple[Tuple[str, str], ComparisonResult]]
):
    # type: (...) -> pd.DataFrame

    dataframeable = []
    for names, result in comparison_results:
        dataframeable.append(dict(
            left_name=names[0],
            right_name=names[1],
            **result._asdict()
        ))

    return pd.DataFrame(dataframeable)

