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
        """Populates a DataObject from an input *.json likely parsed by
        argschema

        Returns
        -------
        DataObject:
            An instantiated DataObject which has `name` and `value` properties
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def to_json(self) -> dict:  # pragma: no cover
        """Given an already populated DataObject, return the dict that
        when used with the `from_json()` classmethod would produce the same
        DataObject

        Returns
        -------
        dict:
            The JSON (in dict form) that would produce the DataObject.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_lims(cls) -> "DataObject":  # pragma: no cover
        """Populate a DataObject from an internal database (likely LIMS)

        Returns
        -------
        DataObject:
            An instantiated DataObject which has `name` and `value` properties
        """
        # Example:
        # return cls(name="my_data_object", value=42)
        raise NotImplementedError()

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

    @abc.abstractmethod
    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:  # pragma: no cover
        """Given an already populated DataObject, return an pyNWB file object
        that had had DataObject data added.

        Parameters
        ----------
        nwbfile : NWBFile
            An NWB file object

        Returns
        -------
        NWBFile
            An NWB file object that has had data from the DataObject added
            to it.
        """
        raise NotImplementedError()
