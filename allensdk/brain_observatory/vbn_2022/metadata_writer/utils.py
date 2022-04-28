from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.vbn_2022.metadata_writer.session_utils import (
    behavior_session_table_from_ecephys_session_id,
    _postprocess_sessions,
    _add_prior_omissions_behavior)


def get_list_of_bad_probe_ids(
        lims_connection: PostgresQueryMixin,
        probes_to_skip: List[Dict[str, Any]]) -> List[int]:
    """
    Given a list of probes to skip (each of the form
    {'session': ecephys_session_id,
     'probe': probe_name}
    return a list of the ecephys_probe_ids associated with
    the bad probes.
    """

    where_clause = ""
    for probe in probes_to_skip:
        if len(where_clause) > 0:
            where_clause += " OR "
        where_clause += f"(ecephys_session_id={probe['session']}"
        where_clause += f" AND name='{probe['probe']}')"

    query = f"""
    SELECT
    id as probe_id
    FROM ecephys_probes
    WHERE
    {where_clause}
    """

    result = lims_connection.fetchall(query)
    return result


def _get_units_table(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will return the units table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    units_table: pd.DataFrame
        A pandas DataFrame corresponding to the units metadata table
    """

    query = """
    select
    ecephys_units.id as unit_id
    ,ecephys_units.ecephys_channel_id
    ,ecephys_probes.id as ecephys_probe_id
    ,ecephys_sessions.id as ecephys_session_id
    ,ecephys_units.snr
    ,ecephys_units.firing_rate
    ,ecephys_units.isi_violations
    ,ecephys_units.presence_ratio
    ,ecephys_units.amplitude_cutoff
    ,ecephys_units.isolation_distance
    ,ecephys_units.l_ratio
    ,ecephys_units.d_prime
    ,ecephys_units.nn_hit_rate
    ,ecephys_units.nn_miss_rate
    ,ecephys_units.silhouette_score
    ,ecephys_units.max_drift
    ,ecephys_units.cumulative_drift
    ,ecephys_units.duration as waveform_duration
    ,ecephys_units.halfwidth as waveform_halfwidth
    ,ecephys_units.\"PT_ratio\" as waveform_PT_ratio
    ,ecephys_units.repolarization_slope as waveform_repolarization_slope
    ,ecephys_units.recovery_slope as waveform_recovery_slope
    ,ecephys_units.amplitude as waveform_amplitude
    ,ecephys_units.spread as waveform_spread
    ,ecephys_units.velocity_above as waveform_velocity_above
    ,ecephys_units.velocity_below as waveform_velocity_below
    ,ecephys_units.local_index
    ,ecephys_channels.probe_vertical_position
    ,ecephys_channels.probe_horizontal_position
    ,ecephys_channels.anterior_posterior_ccf_coordinate
    ,ecephys_channels.dorsal_ventral_ccf_coordinate
    ,ecephys_channels.manual_structure_id as ecephys_structure_id
    ,structures.acronym as ecephys_structure_acronym
    """

    query += """
    FROM ecephys_units
    JOIN ecephys_channels
    ON ecephys_channels.id = ecephys_units.ecephys_channel_id
    JOIN ecephys_probes
    ON ecephys_probes.id = ecephys_channels.ecephys_probe_id
    JOIN ecephys_sessions
    ON ecephys_probes.ecephys_session_id = ecephys_sessions.id
    LEFT JOIN structures
    ON structures.id = ecephys_channels.manual_structure_id
    """

    query += f"""
    WHERE ecephys_sessions.id IN {tuple(session_id_list)}
    """

    if probe_ids_to_skip is not None:
        query += f"""
        AND ecephys_probes.id NOT IN {tuple(probe_ids_to_skip)}
        """

    units_table = lims_connection.select(query)

    return units_table


def _get_probes_table(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will return the probes table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    probes_table: pd.DataFrame
        A pandas DataFrame corresponding to the probes metadata table
    """

    query = """
    select
    ecephys_probes.id as ecephys_probe_id
    ,ecephys_probes.ecephys_session_id
    ,ecephys_probes.name
    ,ecephys_probes.global_probe_sampling_rate as sampling_rate
    ,ecephys_probes.global_probe_lfp_sampling_rate as lfp_sampling_rate
    ,ecephys_probes.phase
    ,ecephys_probes.use_lfp_data as has_lfp_data
    ,count(distinct(ecephys_units.id)) as unit_count
    ,count(distinct(ecephys_channels.id)) as channel_count
    ,array_agg(distinct(structures.acronym)) as ecephys_structure_acronyms"""

    query += """
    FROM  ecephys_probes
    JOIN ecephys_sessions
    ON ecephys_probes.ecephys_session_id = ecephys_sessions.id
    LEFT OUTER JOIN ecephys_channels
    ON ecephys_channels.ecephys_probe_id = ecephys_probes.id
    LEFT OUTER JOIN ecephys_units
    ON ecephys_units.ecephys_channel_id=ecephys_channels.id
    LEFT JOIN structures
    ON structures.id = ecephys_channels.manual_structure_id"""

    query += f"""
    WHERE ecephys_sessions.id IN {tuple(session_id_list)}"""

    if probe_ids_to_skip is not None:
        query += f"""
        AND ecephys_probes.id NOT IN {tuple(probe_ids_to_skip)}
        """

    query += """group by ecephys_probes.id"""

    probes_table = lims_connection.select(query)
    return probes_table


def _get_channels_table(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will return the channels table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    channels_table: pd.DataFrame
        A pandas DataFrame corresponding to the channels metadata table
    """

    query = """
    select
    ecephys_channels.id as ecephys_channel_id
    ,ecephys_channels.ecephys_probe_id
    ,ecephys_sessions.id as ecephys_session_id
    ,ecephys_channels.local_index
    ,ecephys_channels.probe_vertical_position
    ,ecephys_channels.probe_horizontal_position
    ,ecephys_channels.anterior_posterior_ccf_coordinate
    ,ecephys_channels.dorsal_ventral_ccf_coordinate
    ,ecephys_channels.left_right_ccf_coordinate
    ,structures.acronym as ecephys_structure_acronym
    ,count(distinct(ecephys_units.id)) as unit_count
    """

    query += """
    FROM  ecephys_channels
    JOIN ecephys_probes
    ON ecephys_channels.ecephys_probe_id = ecephys_probes.id
    JOIN ecephys_sessions
    ON ecephys_probes.ecephys_session_id = ecephys_sessions.id
    LEFT OUTER JOIN ecephys_units
    ON ecephys_units.ecephys_channel_id=ecephys_channels.id
    LEFT JOIN structures
    ON structures.id = ecephys_channels.manual_structure_id"""

    query += f"""
    WHERE ecephys_sessions.id IN {tuple(session_id_list)}"""

    if probe_ids_to_skip is not None:
        query += f"""
        AND ecephys_probes.id NOT IN {tuple(probe_ids_to_skip)}
        """

    query += """
    GROUP BY ecephys_channels.id,
    ecephys_sessions.id,
    structures.acronym"""

    channels_table = lims_connection.select(query)
    return channels_table


def _get_ecephys_summary_table(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int]) -> pd.DataFrame:
    """
    Perform the database query that will return the session summary table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    Returns
    -------
    sumary_table: pd.DataFrame
        A pandas DataFrame corresponding to the session summary table
    """
    query = """
        SELECT
        ecephys_sessions.id AS ecephys_session_id
        ,behavior_sessions.id as behavior_session_id
        ,ecephys_sessions.date_of_acquisition
        ,equipment.name as equipment_name
        ,ecephys_sessions.stimulus_name as session_type
        ,donors.external_donor_name as mouse_id
        ,donors.full_genotype as genotype
        ,genders.name AS sex
        ,projects.code as project_code
        ,DATE_PART('day',
          ecephys_sessions.date_of_acquisition - donors.date_of_birth)
          AS age_in_days
        """

    query += f"""
        FROM ecephys_sessions
        JOIN specimens
        ON specimens.id = ecephys_sessions.specimen_id
        JOIN donors
        ON specimens.donor_id = donors.id
        JOIN genders
        ON genders.id = donors.gender_id
        JOIN projects
        ON projects.id = ecephys_sessions.project_id
        LEFT OUTER JOIN equipment
        ON equipment.id = ecephys_sessions.equipment_id
        LEFT OUTER JOIN behavior_sessions
        ON behavior_sessions.ecephys_session_id = ecephys_sessions.id
        WHERE ecephys_sessions.id IN {tuple(session_id_list)}"""

    summary_table = lims_connection.select(query)
    return summary_table


def _get_ecephys_counts_per_session(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will produce a table enumerating
    how many units, channels, and probes there are in each
    session

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    counts_table: pd.DataFrame
        A pandas DataFrame corresponding to the counts_per_session
        table
    """

    query = f"""
    SELECT ecephys_sessions.id as ecephys_session_id,
    COUNT(DISTINCT(ecephys_units.id)) as unit_count,
    COUNT(DISTINCT(ecephys_probes.id)) as probe_count,
    COUNT(DISTINCT(ecephys_channels.id)) as channel_count
    FROM ecephys_sessions
    LEFT OUTER JOIN ecephys_probes ON
    ecephys_probes.ecephys_session_id = ecephys_sessions.id
    LEFT OUTER JOIN ecephys_channels ON
    ecephys_channels.ecephys_probe_id = ecephys_probes.id
    LEFT OUTER JOIN ecephys_units ON
    ecephys_units.ecephys_channel_id = ecephys_channels.id
    WHERE ecephys_sessions.id IN {tuple(session_id_list)}
    """

    if probe_ids_to_skip is not None:
        query += f"""
        AND ecephys_probes.id NOT IN {tuple(probe_ids_to_skip)}
        """

    query += """
    GROUP BY ecephys_sessions.id"""

    counts_table = lims_connection.select(query)
    return counts_table


def _get_ecephys_structure_acronyms(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will produce a table listing the
    acronyms of all of the structures encompassed by a given
    session.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    structure_table: pd.DataFrame
        A pandas DataFrame corresponding to the table of
        structure acronyms per session
    """
    query = f"""
    SELECT ecephys_sessions.id as ecephys_session_id,
    array_agg(distinct(structures.acronym)) as ecephys_structure_acronyms
    FROM ecephys_sessions
    JOIN ecephys_probes
    ON ecephys_probes.ecephys_session_id = ecephys_sessions.id
    JOIN ecephys_channels
    ON ecephys_channels.ecephys_probe_id = ecephys_probes.id
    LEFT JOIN structures
    ON structures.id = ecephys_channels.manual_structure_id
    WHERE ecephys_sessions.id IN {tuple(session_id_list)}
    """

    if probe_ids_to_skip is not None:
        query += f"""
        AND ecephys_probes.id NOT IN {tuple(probe_ids_to_skip)}
        """

    query += """
    GROUP BY ecephys_sessions.id"""

    struct_tbl = lims_connection.select(query)
    return struct_tbl


def _add_images_from_behavior(
        ecephys_table: pd.DataFrame,
        behavior_table: pd.DataFrame) -> pd.DataFrame:
    """
    Use the behavior sessions table to add image_set and
    prior_exposurs to image_set to ecephys table
    """
    # add prior exposure to image_set to session_table

    sub_df = behavior_table.loc[
        np.logical_not(behavior_table.ecephys_session_id.isna()),
        ('ecephys_session_id', 'image_set', 'prior_exposures_to_image_set')]

    ecephys_table = ecephys_table.merge(
            sub_df.set_index('ecephys_session_id'),
            on='ecephys_session_id',
            how='left')
    return ecephys_table


def _get_session_tables(
        lims_connection: PostgresQueryMixin,
        mtrain_connection: PostgresQueryMixin,
        session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform the database query to generate the ecephys_session_table

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    mtrain_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    session_table: pd.DataFrame
        A pandas DataFrame corresponding to the session table

    behavior_session-table: pd.DataFrame
    """

    behavior_session_table = behavior_session_table_from_ecephys_session_id(
            lims_connection=lims_connection,
            mtrain_connection=mtrain_connection,
            ecephys_session_ids=session_id_list)

    summary_tbl = _get_ecephys_summary_table(
                        lims_connection=lims_connection,
                        session_id_list=session_id_list)

    ct_tbl = _get_ecephys_counts_per_session(
                        lims_connection=lims_connection,
                        session_id_list=session_id_list,
                        probe_ids_to_skip=probe_ids_to_skip)

    summary_tbl = summary_tbl.join(
                        ct_tbl.set_index("ecephys_session_id"),
                        on="ecephys_session_id",
                        how='left')

    struct_tbl = _get_ecephys_structure_acronyms(
                        lims_connection=lims_connection,
                        session_id_list=session_id_list,
                        probe_ids_to_skip=probe_ids_to_skip)

    summary_tbl = summary_tbl.join(
                     struct_tbl.set_index("ecephys_session_id"),
                     on="ecephys_session_id",
                     how='left')

    summary_tbl = _add_images_from_behavior(
            ecephys_table=summary_tbl,
            behavior_table=behavior_session_table)

    session_table = _postprocess_sessions(
                        sessions_df=summary_tbl)

    # there are only omissions in the ecephys sessions,
    # so behavior.prior_exposures_to_omissions should be zero
    # except for the sessions with non-null ecephys_sessions
    behavior_session_table = _add_prior_omissions_behavior(
            behavior_df=behavior_session_table,
            ecephys_df=session_table)

    behavior_session_table = behavior_session_table[
            ['behavior_session_id',
             'equipment_name',
             'genotype',
             'mouse_id',
             'sex',
             'age_in_days',
             'session_number',
             'prior_exposures_to_session_type',
             'prior_exposures_to_image_set',
             'prior_exposures_to_omissions',
             'ecephys_session_id',
             'date_of_acquisition',
             'session_type',
             'image_set']]

    return (session_table, behavior_session_table)
