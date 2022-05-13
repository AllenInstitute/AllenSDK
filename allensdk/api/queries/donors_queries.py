from typing import List

import pandas as pd

from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.api.queries.utils import build_in_list_selector_query


def get_death_date_for_mouse_ids(
        lims_connections: PostgresQueryMixin,
        mouse_ids_list: List[int]
) -> pd.DataFrame:
    """

    Parameters
    ----------
    lims_connections:
    mouse_ids_list: list of mouse ids

    Returns
    -------
    Dataframe with columns:
        - mouse id: int
        - death on: datetime
    """
    query = f"""
        SELECT external_donor_name as mouse_id, death_on
        FROM donors
        {build_in_list_selector_query(
        col='external_donor_name',
        valid_list=mouse_ids_list
    )}
    """
    return lims_connections.select(query=query)
