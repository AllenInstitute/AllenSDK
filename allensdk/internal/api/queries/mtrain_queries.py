from typing import Optional, List
import pandas as pd
from allensdk.internal.api import PostgresQueryMixin
from allensdk.internal.api.queries.utils import (
    build_in_list_selector_query)
import logging


def session_stage_from_foraging_id(
        mtrain_engine: PostgresQueryMixin,
        foraging_ids: Optional[List[str]] = None,
        logger: Optional[logging.RootLogger] = None) -> pd.DataFrame:
    """
    Get DataFrame mapping behavior_sessions.id to session_type
    by querying mtrain for foraging_ids
    """
    # Select fewer rows if possible via behavior_session_id
    if foraging_ids is not None:
        foraging_ids = [f"'{fid}'" for fid in foraging_ids]
    # Otherwise just get the full table from mtrain
    else:
        foraging_ids = None

    foraging_ids_query = build_in_list_selector_query(
        "bs.id", foraging_ids)

    query = f"""
        SELECT
            stages.name as session_type,
            bs.id AS foraging_id
        FROM behavior_sessions bs
        JOIN stages ON stages.id = bs.state_id
        {foraging_ids_query};
        """
    if logger is not None:
        logger.debug(f"_get_behavior_stage_table query: \n {query}")
    return mtrain_engine.select(query)
