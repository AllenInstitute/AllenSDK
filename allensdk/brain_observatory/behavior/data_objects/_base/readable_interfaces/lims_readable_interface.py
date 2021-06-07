import abc

from allensdk.brain_observatory.behavior.data_objects import DataObject


class LimsReadableInterface(abc.ABC):
    """Marks a data object as readable from LIMS"""
    @classmethod
    @abc.abstractmethod
    def from_lims(cls, *args) -> "DataObject":  # pragma: no cover
        """Populate a DataObject from an internal database (likely LIMS)

        Returns
        -------
        DataObject:
            An instantiated DataObject which has `name` and `value` properties
        """
        # Example:
        # return cls(name="my_data_object", value=42)
        raise NotImplementedError()
