from typing import Optional, List
import pandas as pd
from uuid import UUID
from allensdk.internal.api import PostgresQueryMixin
from allensdk.core.typing import SupportsStr
import logging


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


def foraging_id_map_from_behavior_session_id(
        lims_engine: PostgresQueryMixin,
        behavior_session_ids: List[int],
        logger: Optional[logging.RootLogger] = None) -> pd.DataFrame:
    """
    Returns DataFrame with two columns:
    foraging_id
    behavior_session_id
    """

    behav_ids = build_in_list_selector_query("id",
                                             behavior_session_ids,
                                             operator="AND")
    forag_ids_query = f"""
            SELECT foraging_id, id as behavior_session_id
            FROM behavior_sessions
            WHERE foraging_id IS NOT NULL
            {behav_ids};
            """
    if logger is not None:
        logger.debug("get_foraging_ids_from_behavior_session query: \n"
                     f"{forag_ids_query}")
    foraging_id_map = lims_engine.select(forag_ids_query)

    if logger is not None:
        logger.debug(f"Retrieved {len(foraging_id_map)} foraging ids for"
                     " behavior stage query. "
                     f"Ids = {foraging_id_map.foraging_id}")
    return foraging_id_map


def _sanitize_uuid_list(uuid_list: List[str]) -> List[str]:
    """
    Loop over a string, removing any that cannot be cast
    into a valid UUID
    """
    new_list = []
    for val in uuid_list:
        try:
            UUID(val)
            new_list.append(val)
        except ValueError:
            pass
    return new_list
