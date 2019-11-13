import re
from typing import Union
import difflib


class WhitespaceStrippedString(object):
    """Comparator class to compare strings that have been stripped of
    whitespace. By default removes any unicode whitespace character that
    matches the regex \s, (which includes [ \t\n\r\f\v], and other unicode
    whitespace characters).
    """
    def __init__(self, string: str, whitespace_chars: str = r"\s",
                 ASCII: bool = False):
        self.orig = string
        self.whitespace_chars = whitespace_chars
        self.flags = re.ASCII if ASCII else 0
        self.differ = difflib.Differ()
        self.value = re.sub(self.whitespace_chars, "", string, self.flags)

    def __eq__(self, other: Union[str, "WhitespaceStrippedString"]):
        if isinstance(other, str):
            other = WhitespaceStrippedString(
                other, self.whitespace_chars, self.flags)
        self.diff = list(self.differ.compare(self.value, other.value))
        return self.value == other.value
