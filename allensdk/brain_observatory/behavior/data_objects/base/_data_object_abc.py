import abc
from collections import deque
from typing import Any, Optional, Set

from pynwb import NWBFile

from allensdk.brain_observatory.comparison_utils import compare_fields


class DataObject(abc.ABC):
    """An abstract class that prototypes properties that represent a
    category of experimental data/metadata (e.g. running speed,
    rewards, licks, etc.) and that prototypes methods to allow conversion of
    the experimental data/metadata to and from various
    data sources and sinks (e.g. LIMS, JSON, NWB).
    """

    def __init__(self, name: str, value: Any,
                 exclude_from_equals: Optional[Set] = None):
        """
        :param name
            Name
        :param value
            Value
        :param exclude_from_equals
            Optional set which will exclude these properties from comparison
            checks to another DataObject
        """
        self._name = name
        self._value = value

        efe = exclude_from_equals if exclude_from_equals else set()
        self._exclude_from_equals = efe

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Any:
        return self._value

    @classmethod
    def _get_vars(cls):
        return vars(cls)

    def to_dict(self):
        res = {}
        q = deque([(name, value, []) for name, value in
                   self._get_properties().items()])
        while q:
            name, value, path = q.popleft()
            if isinstance(value, DataObject):
                cur = res
                for p in path:
                    cur = cur[p]
                cur[name] = {}
                newpath = path + [name]
                values = [(name, value, newpath) for name, value in
                          value._get_properties().items()]
                if not values:
                    q.append((name, value._value, newpath))
                else:
                    for v in values:
                        q.append(v)
            else:
                cur = res
                for p in path:
                    cur = cur[p]
                cur[name] = value

        return res

    def _get_properties(self):
        """Returns all property names and values"""
        vars_ = self._get_vars()
        return {name: getattr(self, name) for name, value in vars_.items()
                if isinstance(value, property)}

    def __eq__(self, other: "DataObject"):
        if type(self) != type(other):
            msg = f'Do not know how to compare with type {type(other)}'
            raise NotImplementedError(msg)

        d_self = self.to_dict()
        d_other = other.to_dict()

        # if not isinstance(self._value, DataObject):
        #     properties_self['value'] = self._value
        #     properties_other['value'] = other._value
        #
        # properties_self['name'] = self._name
        # properties_other['name'] = other._name

        for p in d_self:
            if p in self._exclude_from_equals:
                continue

            x1 = d_self[p]
            x2 = d_other[p]

            try:
                compare_fields(x1=x1, x2=x2)
            except AssertionError:
                return False
        return True
