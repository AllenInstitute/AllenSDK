from typing import _Protocol
from abc import abstractmethod


class SupportsStr(_Protocol):
    """Classes that support the __str__ method"""
    @abstractmethod
    def __str__(self) -> str:
        pass
