import sys
if sys.version_info.minor <= 7:  
    from typing import _Protocol as Protocol
else:
    from typing import Protocol
from abc import abstractmethod


class SupportsStr(Protocol):
    """Classes that support the __str__ method"""
    @abstractmethod
    def __str__(self) -> str:
        pass
