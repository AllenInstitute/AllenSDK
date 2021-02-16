import abc


class AbstractDataObject(abc.ABC):
    @abc.abstractmethod
    def from_lims():
        raise NotImplementedError()

    @abc.abstractmethod
    def to_dict():
        raise NotImplementedError()
