import abc

from pynwb import NWBFile


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


class NwbWritableInterface(abc.ABC):
    """Marks a data object as writable to NWB"""
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
