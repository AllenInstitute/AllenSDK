from typing import List
from allensdk.internal.api import PostgresQueryMixin


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
