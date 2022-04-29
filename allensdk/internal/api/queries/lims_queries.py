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

    Parameters
    ----------
    lims_engine: PostgresQueryMixin
        Means of connecting to the LIMS database

    behavior_session_ids: List[int]
        List of behavior_session_ids for which we want
        the foraging_id
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


def donor_id_list_from_ecephys_session_ids(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int]) -> List[int]:
    """
    Get the list of donor IDs associated with a list
    of ecephys_session_ids
    """
    query = f"""
    SELECT DISTINCT(donors.id) as donor_id
    FROM donors
    JOIN specimens ON
    specimens.donor_id = donors.id
    JOIN ecephys_sessions ON
    ecephys_sessions.specimen_id = specimens.id
    WHERE
    ecephys_sessions.id in {tuple(session_id_list)}
    """
    result = lims_connection.select(query)
    return list(result.donor_id)


def behavior_sessions_from_ecephys_session_ids(
        lims_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int]) -> pd.DataFrame:
    """
    Get a DataFrame listing all of the behavior sessions that
    mice from a specified list of ecephys sessions went through

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]
        The ecephys sessions used to find the mice used to find
        the behavior sessions

    Returns
    -------
    mouse_to_behavior: pd.DataFrame
        Dataframe with columns
            mouse_id
            behavior_session_id
            date_of_acquisition
        listing every behavior session the mice in question went through
    """
    donor_id_list = donor_id_list_from_ecephys_session_ids(
                        lims_connection=lims_connection,
                        session_id_list=ecephys_session_id_list)

    query = f"""
    SELECT
    donors.external_donor_name as mouse_id
    ,behavior.id as behavior_session_id
    ,behavior.date_of_acquisition as date_of_acquisition
    FROM donors
    JOIN behavior_sessions AS behavior
    ON behavior.donor_id = donors.id
    WHERE
    donors.id in {tuple(donor_id_list)}
    """
    mouse_to_behavior = lims_connection.select(query)
    return mouse_to_behavior


def stimulus_pickle_paths_from_behavior_session_ids(
        lims_connection: PostgresQueryMixin,
        behavior_session_id_list: List[int]) -> pd.DataFrame:
    """
    Get a DataFrame mapping behavior_session_id to
    stimulus_pickle_path

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    behavior_session_id_list: List[int]

    Returns
    -------
    beh_to_path: pd.DataFrame
        with columns
            behavior_session_id
            pkl_path
    """

    query = f"""
    SELECT
    beh.id as behavior_session_id
    ,wkf.storage_directory || wkf.filename as pkl_path
    FROM behavior_sessions AS beh
    JOIN
    well_known_files AS wkf
    ON wkf.attachable_id = beh.id
    JOIN
    well_known_file_types as wkft
    ON
    wkf.well_known_file_type_id = wkft.id
    WHERE
    wkft.name = 'StimulusPickle'
    AND
    wkf.attachable_type = 'BehaviorSession'
    AND
    beh.id in {tuple(behavior_session_id_list)}
    """

    beh_to_path = lims_connection.select(query)
    return beh_to_path
