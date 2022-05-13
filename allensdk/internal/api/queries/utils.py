from typing import Optional, List
from allensdk.core.typing import SupportsStr
from uuid import UUID


def build_in_list_selector_query(
        col: str,
        valid_list: Optional[SupportsStr] = None,
        operator: str = "WHERE",
        valid: bool = True) -> str:
    """
    Filter for rows where the value of a column is contained in a list
    (or, if valid=False, where the value is not contained in that list).
    If no list is specified in `valid_list`, return an empty string.

    NOTE: if string ids are used, then the strings in `valid_list` must
    be enclosed in single quotes, or else the query will throw a column
    does not exist error. E.g. ["'mystringid1'", "'mystringid2'"...]

    Parameters
    ----------
    col: str
        The name of the column being filtered on

    valid_list: Optional[SupportsStr]
        The list of values to test column on

    operator: str
        The SQL operator that starts the clause ("WHERE", "AND" or "OR")

    valid: bool
        If True, test for "col IN valid_list"; else, test for
        "col NOT IN valid_list"

    Returns
    -------
    session_query: str
        The clause performing the request filter
    """
    if operator not in ("AND", "OR", "WHERE"):
        msg = ("Operator must be 'AND', 'OR', or 'WHERE'; "
               f"you gave '{operator}'")
        raise ValueError(msg)

    if not valid_list:
        return ""

    if valid:
        relation = "IN"
    else:
        relation = "NOT IN"

    session_query = (
        f"""{operator} {col} {relation} ({",".join(
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
