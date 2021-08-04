import inspect
import logging
import math
import warnings
import datetime

from itertools import zip_longest
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr
import SimpleITK as sitk

from pandas.util.testing import assert_frame_equal

from allensdk.core.lazy_property import LazyProperty

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_equal(a: Any, b: Any) -> bool:
    """Function to deal with checking if two variables of possibly mixed types
    have the same value."""

    if type(a) != type(b):
        return False

    if isinstance(a, (pd.Series, pd.DataFrame)):
        return a.equals(b)
    elif isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    elif isinstance(a, (list, tuple)):
        for a_elem, b_elem in zip_longest(a, b):
            if not is_equal(a_elem, b_elem):
                return False
        return True
    elif isinstance(a, set):
        for a_elem, b_elem in zip_longest(sorted(a), sorted(b)):
            if not is_equal(a_elem, b_elem):
                return False
        return True
    elif isinstance(a, dict):
        for (a_k, a_v), (b_k, b_v) in zip_longest(sorted(a.items()),
                                                  sorted(b.items())):
            if (a_k != b_k) or (not is_equal(a_v, b_v)):
                return False
        return True
    else:
        return bool(a == b)


class ParamsMixin:
    """This mixin adds parameter management functionality to the class it is
    mixed into.

    This mixin expects that the class it is mixed into will have an __init__
    with type annotated parameters. It also expects for the class to have
    semi-private attributes of the __init__ type annotated parameters.

    Example:

    SomeClassWhereParamManagementIsDesired(ParamsMixin):

        # Managed params should be typed (with simple types if possible)!
        def __init__(self, param_to_ignore, a_param_1: int, a_param_2: float,
                     b_param_1: list):
            # Parameters can be ignored by the mixin
            super().__init__(ignore={'param_to_ignore'})

            # Pay attention to the naming scheme!
            self._a_param_1 = a_param_1
            self._a_param_2 = a_param_2
            self._b_param_1 = b_param_1

        ...

    After being mixed in, methods like 'get_params', 'set_params',
    'needs_data_refresh', and 'clear_updated_params' will be available.
    """

    def __init__(self, ignore: set = {'api'}):
        self._updated_params: set = set()
        self._ignore = ignore

    @classmethod
    def _get_param_signatures(cls) -> List[inspect.Parameter]:
        init = getattr(cls, '__init__')
        if init is object.__init__:
            # Class has a default __init__ and thus no params
            return []
        init_signature = inspect.signature(init)
        # Filter out 'self' and '**kwargs' params
        parameters = [p for p in init_signature.parameters.values()
                      if (p.name != 'self') and (p.kind != p.VAR_KEYWORD)]
        return parameters

    @classmethod
    def _get_param_type_annotations(cls) -> Dict[str, type]:
        parameters = cls._get_param_signatures()
        return {p.name: p.annotation for p in parameters}

    @classmethod
    def _get_param_names(cls) -> List[str]:
        parameters = cls._get_param_signatures()
        return sorted([p.name for p in parameters])

    def get_params(self) -> Dict[str, Any]:
        """Get managed params and their values"""
        out = dict()
        for param in self._get_param_names():
            if param in self._ignore:
                continue
            value = getattr(self, f"_{param}")
            out.update({param: value})
        return out

    def set_params(self, **params):
        """Set managed params"""
        valid_params = self.get_params().keys()
        param_types = self._get_param_type_annotations()
        current_params = self.get_params()

        for param, value in params.items():
            if param in valid_params:
                current_value = current_params[param]

                if isinstance(value, param_types[param]):
                    if not is_equal(current_value, value):
                        setattr(self, f"_{param}", value)
                        self._updated_params.add(param)
                else:
                    warnings.warn(f"The value ({value}) for parameter "
                                  f"'{param}' should be of type "
                                  f"'{param_types[param]}' but is instead "
                                  f"{type(value)}. It will remain as: "
                                  f"{current_value} "
                                  f"({type(current_value)}).",
                                  stacklevel=2)
            else:
                warnings.warn(f"The parameter '{param}' is not valid "
                              f"and is being ignored! "
                              f"Possible params are: {valid_params}",
                              stacklevel=2)

    def needs_data_refresh(self, data_params: set) -> bool:
        """Check if specific params have been updated via `set_params()`"""
        return bool(data_params & self._updated_params)

    def clear_updated_params(self, data_params: set):
        """This method clears 'updated params' whose data have been updated"""
        self._updated_params -= data_params


def sessions_are_equal(A, B, reraise=False) -> bool:
    """Check if two Session objects are equal (have same methods and
    attributes).

    Parameters
    ----------
    A : Session A
        The first session to compare
    B : Session B
        The second session to compare
    reraise : bool, optional
        Whether to reraise when encountering an Assertion or AttributeError,
        by default False

    Returns
    -------
    bool
        Whether the two sessions are equal to one another.
    """

    field_set = set()
    for key, val in A.__dict__.items():
        if isinstance(val, LazyProperty):
            field_set.add(key)
    for key, val in B.__dict__.items():
        if isinstance(val, LazyProperty):
            field_set.add(key)

    logger.info(f"Comparing the following fields: {field_set}")

    for field in sorted(field_set):
        try:
            logger.info(f"Comparing field: {field}")
            x1, x2 = getattr(A, field), getattr(B, field)
            err_msg = (f"{field} on {A} did not equal {field} "
                       f"on {B} (\n{x1} vs\n{x2}\n)")
            compare_session_fields(x1, x2, err_msg)

        except NotImplementedError:
            A_implements_get_field = hasattr(
                A.api, getattr(type(A), field).getter_name)
            B_implements_get_field = hasattr(
                B.api, getattr(type(B), field).getter_name)
            assert ((A_implements_get_field is False)
                    and (B_implements_get_field is False))

        except (AssertionError, AttributeError):
            if reraise:
                raise
            return False

    return True


def compare_session_fields(x1: Any, x2: Any, err_msg=""):
    """Helper function to compare if two fields (attributes) from a
    Session object are equal to one another.

    Parameters
    ----------
    x1 : Any
        The field from the first session to compare
    x2 : Any
        The corresponding field from the second session to compare
    err_msg : str, optional
        The error message to display if two compared fields do not equal
        one another, by default "" (an empty string)
    """
    if isinstance(x1, pd.DataFrame):
        try:
            assert_frame_equal(x1, x2, check_like=True)
        except Exception:
            print(err_msg)
            raise
    elif isinstance(x1, np.ndarray):
        np.testing.assert_array_almost_equal(x1, x2, err_msg=err_msg)
    elif isinstance(x1, xr.DataArray):
        xr.testing.assert_allclose(x1, x2)
    elif isinstance(x1, (list,)):
        assert x1 == x2, err_msg
    elif isinstance(x1, (sitk.Image,)):
        assert x1.GetSize() == x2.GetSize(), err_msg
        assert x1 == x2, err_msg
    elif isinstance(x1, (datetime.datetime, pd.Timestamp)):
        if isinstance(x1, pd.Timestamp):
            x1 = x1.to_pydatetime()
        if isinstance(x2, pd.Timestamp):
            x2 = x2.to_pydatetime()
        time_delta = (x1 - x2).total_seconds()
        # Timestamp differences should be less than 60 seconds
        assert abs(time_delta) < 60
    elif isinstance(x1, (float,)):
        if math.isnan(x1) or math.isnan(x2):
            both_nan = (math.isnan(x1) and math.isnan(x2))
            assert both_nan, err_msg
        else:
            assert x1 == x2, err_msg
    elif isinstance(x1, (dict,)):
        for key in set(x1.keys()).union(set(x2.keys())):
            key_err_msg = f"Mismatch when checking key {key}. {err_msg}"
            compare_session_fields(x1[key], x2[key], err_msg=key_err_msg)
    else:
        assert x1 == x2, err_msg
