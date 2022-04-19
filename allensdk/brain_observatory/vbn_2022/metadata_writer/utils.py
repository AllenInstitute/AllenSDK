from typing import List
import pandas as pd
from allensdk.internal.api import PostgresQueryMixin


def _get_units_table(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int]) -> pd.DataFrame:
    """
    Perform the database query that will return the units table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    Returns
    -------
    units_table: pd.DataFrame
        A pandas DataFrame corresponding to the units metadata table
    """

    query = """
    select
    eu.id as unit_id
    ,eu.ecephys_channel_id
    ,ep.id as ecephys_probe_id
    ,es.id as ecephys_session_id
    ,eu.snr
    ,eu.firing_rate
    ,eu.isi_violations
    ,eu.presence_ratio
    ,eu.amplitude_cutoff
    ,eu.isolation_distance
    ,eu.l_ratio
    ,eu.d_prime
    ,eu.nn_hit_rate
    ,eu.nn_miss_rate
    ,eu.silhouette_score
    ,eu.max_drift
    ,eu.cumulative_drift
    ,eu.duration as waveform_duration
    ,eu.halfwidth as waveform_halfwidth
    ,eu.\"PT_ratio\" as waveform_PT_ratio
    ,eu.repolarization_slope as waveform_repolarization_slope
    ,eu.recovery_slope as waveform_recovery_slope
    ,eu.amplitude as waveform_amplitude
    ,eu.spread as waveform_spread
    ,eu.velocity_above as waveform_velocity_above
    ,eu.velocity_below as waveform_velocity_below
    ,eu.local_index
    ,ec.probe_vertical_position
    ,ec.probe_horizontal_position
    ,ec.anterior_posterior_ccf_coordinate
    ,ec.dorsal_ventral_ccf_coordinate
    ,ec.manual_structure_id as ecephys_structure_id
    ,st.acronym as ecephys_structure_acronym
    """

    query += """
    FROM ecephys_units as eu
    JOIN ecephys_channels as ec on ec.id = eu.ecephys_channel_id
    JOIN ecephys_probes as ep on ep.id = ec.ecephys_probe_id
    JOIN ecephys_sessions as es on ep.ecephys_session_id = es.id
    LEFT JOIN structures as st on st.id = ec.manual_structure_id
    """

    query += f"""
    WHERE es.id IN {tuple(session_id_list)}
    """
    units_table = lims_connection.select(query)

    return units_table


def _get_probes_table(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int]) -> pd.DataFrame:
    """
    Perform the database query that will return the probes table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    Returns
    -------
    probes_table: pd.DataFrame
        A pandas DataFrame corresponding to the probes metadata table
    """

    query = """
    select
    ep.id as ecephys_probe_id
    ,ep.ecephys_session_id
    ,ep.name
    ,ep.global_probe_sampling_rate as sampling_rate
    ,ep.global_probe_lfp_sampling_rate as lfp_sampling_rate
    ,ep.phase
    ,ep.use_lfp_data as has_lfp_data
    ,count(distinct(eu.id)) as unit_count
    ,count(distinct(ec.id)) as channel_count
    ,array_agg(distinct(st.acronym)) as ecephys_structure_acronyms"""

    query += """
    FROM  ecephys_probes as ep
    JOIN ecephys_sessions as es on ep.ecephys_session_id = es.id
    LEFT OUTER JOIN ecephys_channels as ec on ec.ecephys_probe_id = ep.id
    LEFT OUTER JOIN ecephys_units as eu on eu.ecephys_channel_id=ec.id
    LEFT JOIN structures st on st.id = ec.manual_structure_id"""

    query += f"""
    WHERE es.id in {tuple(session_id_list)}"""

    query += """group by ep.id"""

    probes_table = lims_connection.select(query)
    return probes_table


def _get_channels_table(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int]) -> pd.DataFrame:
    """
    Perform the database query that will return the channels table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    Returns
    -------
    channels_table: pd.DataFrame
        A pandas DataFrame corresponding to the channels metadata table
    """

    query = """
    select
    ec.id as ecephys_channel_id
    ,ec.ecephys_probe_id
    ,es.id as ecephys_session_id
    ,ec.local_index
    ,ec.probe_vertical_position
    ,ec.probe_horizontal_position
    ,ec.anterior_posterior_ccf_coordinate
    ,ec.dorsal_ventral_ccf_coordinate
    ,ec.left_right_ccf_coordinate
    ,st.acronym as ecephys_structure_acronym
    ,count(distinct(eu.id)) as unit_count
    """

    query += """
    FROM  ecephys_channels as ec
    JOIN ecephys_probes as ep on ec.ecephys_probe_id = ep.id
    JOIN ecephys_sessions as es on ep.ecephys_session_id = es.id
    LEFT OUTER JOIN ecephys_units as eu on eu.ecephys_channel_id=ec.id
    LEFT JOIN structures st on st.id = ec.manual_structure_id"""

    query += f"""
    WHERE es.id in {tuple(session_id_list)}"""

    query += """group by ec.id, es.id, st.acronym"""

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
        es.id AS ecephys_session_id
        ,bs.id as behavior_session_id
        ,es.date_of_acquisition
        ,equipment.name as equipment_name
        ,es.stimulus_name as session_type
        ,d.external_donor_name as mouse_id
        ,d.full_genotype as genotype
        ,g.name AS sex
        ,pr.code as project_code
        ,DATE_PART('day', es.date_of_acquisition - d.date_of_birth)
              AS age_in_days
        """

    query += f"""
        FROM ecephys_sessions as es
        JOIN specimens s on s.id = es.specimen_id
        JOIN donors d on s.donor_id = d.id
        JOIN genders g on g.id = d.gender_id
        JOIN projects pr on pr.id = es.project_id
        LEFT OUTER JOIN equipment on equipment.id = es.equipment_id
        LEFT OUTER JOIN behavior_sessions bs on bs.ecephys_session_id = es.id
        WHERE es.id in {tuple(session_id_list)}"""

    summary_table = lims_connection.select(query)
    return summary_table


def _get_ecephys_counts_per_session(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int]) -> pd.DataFrame:
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
    WHERE ecephys_sessions.id in {tuple(session_id_list)}
    GROUP BY ecephys_sessions.id"""

    counts_table = lims_connection.select(query)
    return counts_table


def _get_ecephys_structure_acronyms(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int]) -> pd.DataFrame:
    """
    Perform the database query that will producea table listing the
    acronyms of all of the structures encompassed by a given
    session.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

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
    WHERE ecephys_sessions.id in {tuple(session_id_list)}
    GROUP BY ecephys_sessions.id"""

    struct_tbl = lims_connection.select(query)
    return struct_tbl


def _get_donor_id_list_from_ecephys_sessions(
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


def _get_behavior_sessions_from_ecephys_sessions(
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
    donor_id_list = _get_donor_id_list_from_ecephys_sessions(
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


def _get_ecephys_session_table(
        lims_connection: PostgresQueryMixin,
        session_id_list: List[int]) -> pd.DataFrame:
    """
    Perform the database query to generate the ecephys_session_table

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    Returns
    -------
    session_table: pd.DataFrame
        A pandas DataFrame corresponding to the session table
    """

    summary_tbl = _get_ecephys_summary_table(
                        lims_connection=lims_connection,
                        session_id_list=session_id_list)

    ct_tbl = _get_ecephys_counts_per_session(
                        lims_connection=lims_connection,
                        session_id_list=session_id_list)

    summary_tbl = summary_tbl.join(
                        ct_tbl.set_index("ecephys_session_id"),
                        on="ecephys_session_id",
                        how='left')

    struct_tbl = _get_ecephys_structure_acronyms(
                        lims_connection=lims_connection,
                        session_id_list=session_id_list)

    summary_tbl = summary_tbl.join(
                     struct_tbl.set_index("ecephys_session_id"),
                     on="ecephys_session_id",
                     how='left')

    return summary_tbl
