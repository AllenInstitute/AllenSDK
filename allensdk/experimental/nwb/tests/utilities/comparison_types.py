from collections import namedtuple
from typing import Callable, Dict, Tuple, Any, List, Union

import pynwb
import numpy as np


StackItemType = Tuple[str, Any]

ComparatorType = Callable[[Any, Any], Tuple[bool, str]]
ComparatorLookupType = Callable[[type, type], ComparatorType]

StackerType = Callable[[Any, Any], Tuple[List[StackItemType], List[StackItemType]]]
StackerLookupType = Callable[[type, type], Union[StackerType, None]]

TypeComparatorType = Callable[[type, type], Tuple[bool, str]]


integer_types = (int, np.integer, bool, np.bool_)
float_types = (float, np.floating)
scalar_numeric_types = integer_types + float_types
list_like_types = (list, np.ndarray, tuple, pynwb.NWBData)
catchall_types = (type(None), str, set)

ComparisonResult = namedtuple('ComparisonResult', ['types_equal', 'direct_equal', 'types_diff', 'direct_diff'])
