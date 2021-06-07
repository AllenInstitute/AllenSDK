import abc

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject


class JsonWritableInterface(abc.ABC):
    """Marks a data object as writable to NWB"""
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
