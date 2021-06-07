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
    def from_nwb(cls, nwbfile: NWBFile) -> "DataObject":  # pragma: no cover
        """Populate a DataObject from a pyNWB file object.

        Parameters
        ----------
        nwbfile:
            The file object (NWBFile) of a pynwb dataset file.

        Returns
        -------
        DataObject:
            An instantiated DataObject which has `name` and `value` properties
        """
        raise NotImplementedError()
