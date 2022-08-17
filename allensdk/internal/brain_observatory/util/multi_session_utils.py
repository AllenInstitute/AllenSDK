"""Utilities for accessing data across multiple sessions"""
import os
from multiprocessing import Pool
from typing import List, Optional

from tqdm import tqdm

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.internal.api import PostgresQueryMixin


def get_session_types_multiprocessing(
        behavior_session_ids: List[int],
        lims_engine: PostgresQueryMixin,
        n_workers: Optional[int] = None
) -> List[str]:
    """Gets session type for `behavior_session_ids` by reading it from the
    behavior stimulus pkl files. Uses multiprocessing to speed up
    reading

    Parameters
    ----------
    behavior_session_ids
        behavior session ids to fetch session type for
    lims_engine
        connection to lims DB
    n_workers
        Number of processes to use. If None, will use all available
        cores

    Returns
    -------
    List[str]: list of session type
    """
    if n_workers is None:
        n_workers = len(os.sched_getaffinity(0))

    with Pool(n_workers) as p:
        stimulus_names = list(tqdm(
            p.imap(_get_session_type_from_pkl_file,
                   zip(
                       behavior_session_ids,
                       [lims_engine] * len(behavior_session_ids))
                   ),
            total=len(behavior_session_ids),
            desc='Reading session type from pkl file'))
    return stimulus_names


def get_session_type_from_pkl_file(
        behavior_session_id: int,
        db_conn: PostgresQueryMixin):
    return _get_session_type_from_pkl_file(behavior_session_id, db_conn)


def _get_session_type_from_pkl_file(*args) -> dict:
    """
    Helper function to get session type from behavior stimulus file
    """
    behavior_session_id, db_conn = args[0]
    return {
        'behavior_session_id': behavior_session_id,
        'session_type': (
            BehaviorStimulusFile.from_lims(
                db=db_conn,
                behavior_session_id=behavior_session_id).session_type)
    }
