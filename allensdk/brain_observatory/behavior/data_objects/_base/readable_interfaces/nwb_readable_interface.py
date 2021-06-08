import abc

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject


class NwbReadableInterface(abc.ABC):
    """Marks a data object as readable from NWB"""
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
