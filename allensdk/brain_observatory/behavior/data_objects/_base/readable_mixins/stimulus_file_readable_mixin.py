import abc

from allensdk.brain_observatory.behavior.data_objects import DataObject


class StimulusFileReadableMixin:
    """Marks a data object as readable from stimulus file"""
    @classmethod
    @abc.abstractmethod
    def from_stimulus_file(cls, *args) \
            -> "DataObject":  # pragma: no cover
        """Populate a DataObject from the stimulus file

        Returns
        -------
        DataObject:
            An instantiated DataObject which has `name` and `value` properties
        """
        raise NotImplementedError()
