from typing import Optional, List
from allensdk.core.typing import SupportsStr
from uuid import UUID


def build_in_list_selector_query(
        col,
        valid_list: Optional[SupportsStr] = None,
        operator: str = "WHERE") -> str:
    """
    Filter for rows where the value of a column is contained in a list.
    If no list is specified in `valid_list`, return an empty string.

    NOTE: if string ids are used, then the strings in `valid_list` must
    be enclosed in single quotes, or else the query will throw a column
    does not exist error. E.g. ["'mystringid1'", "'mystringid2'"...]

    :param col: name of column to compare if in a list
    :type col: str
    :param valid_list: iterable of values that can be mapped to str
        (e.g. string, int, float).
    :type valid_list: list
    :param operator: SQL operator to start the clause. Default="WHERE".
        Valid inputs: "AND", "OR", "WHERE" (not case-sensitive).
    :type operator: str
    """
    if not valid_list:
        return ""
    session_query = (
        f"""{operator} {col} IN ({",".join(
            sorted(set(map(str, valid_list))))})""")
    return session_query


def _sanitize_uuid_list(uuid_list: List[str]) -> List[str]:
    """
    Loop over a list of strings, removing any that cannot be cast
    into a valid UUID

    Parameters
    ----------
    uuid_list: List[str]
        List of strings that would ideally be cast into
        UUIDs

    Returns
    -------
    sanitized_list: List[str]
        A list containing all of the elements from uuid_list
        that could successfully be cast into a UUID

    Note
    ----
    This method is meant to be used as a pre-processing step
    for queries to MTRAIN. foraging_id values need to be valid
    string representations of UUIDs.
    """
    sanitized_list = []
    for val in uuid_list:
        try:
            UUID(val)
            sanitized_list.append(val)
        except ValueError:
            pass
    return sanitized_list
