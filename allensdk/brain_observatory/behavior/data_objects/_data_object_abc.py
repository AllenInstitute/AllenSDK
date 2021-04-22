import abc
from typing import Any

from pynwb import NWBFile


class DataObject(abc.ABC):
    """An abstract class that prototypes properties that represent a
    category of experimental data/metadata (e.g. running speed,
    rewards, licks, etc.) and that prototypes methods to allow conversion of
    the experimental data/metadata to and from various
    data sources and sinks (e.g. LIMS, JSON, NWB).
    """

    def __init__(self, name: str, value: Any):
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Any:
        return self._value

    @classmethod
    @abc.abstractmethod
    def from_json(cls) -> "DataObject":  # pragma: no cover
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_lims(cls) -> "DataObject":  # pragma: no cover
        # Example:
        # return cls(name="my_data_object", value=42)
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "DataObject":  # pragma: no cover
        raise NotImplementedError()

    @abc.abstractmethod
    def to_json(self) -> dict:  # pragma: no cover
        raise NotImplementedError()

    @abc.abstractmethod
    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:  # pragma: no cover
        raise NotImplementedError()
