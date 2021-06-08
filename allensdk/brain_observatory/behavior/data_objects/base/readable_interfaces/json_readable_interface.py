import abc

from allensdk.brain_observatory.behavior.data_objects import DataObject


class JsonReadableInterface(abc.ABC):
    """Marks a data object as readable from json"""
    @classmethod
    @abc.abstractmethod
    def from_json(cls, dict_repr: dict) -> "DataObject":  # pragma: no cover
        """Populates a DataFile from a JSON compatible dict (likely parsed by
        argschema)

        Returns
        -------
        DataObject:
            An instantiated DataObject which has `name` and `value` properties
        """
        raise NotImplementedError()
