import abc

from allensdk.brain_observatory.behavior.data_objects import DataObject


class InternalReadableInterface(abc.ABC):
    """Marks a data object as readable from a variety of internal data sources
    """
    @classmethod
    @abc.abstractmethod
    def from_internal(cls, *args) -> "DataObject":  # pragma: no cover
        """Populate a DataObject from various internal data sources

        Returns
        -------
        DataObject:
            An instantiated DataObject which has `name` and `value` properties
        """
        # Example:
        # return cls(name="my_data_object", value=42)
        raise NotImplementedError()
