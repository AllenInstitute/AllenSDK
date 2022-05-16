from typing import List, Tuple, Dict, Any, Optional
import pandas as pd

from allensdk.api.queries.donors_queries import get_death_date_for_mouse_ids
from allensdk.internal.api import PostgresQueryMixin

from allensdk.internal.api.queries.utils import (
    _sanitize_uuid_list,
    build_in_list_selector_query)

from allensdk.internal.api.queries.behavior_lims_queries import (
    foraging_id_map_from_behavior_session_id)

from allensdk.internal.api.queries.compound_lims_queries import (
    behavior_sessions_from_ecephys_session_ids)

from allensdk.internal.api.queries.mtrain_queries import (
    session_stage_from_foraging_id)

from allensdk.core.dataframe_utils import (
    patch_df_from_other)

from allensdk.brain_observatory.vbn_2022. \
    metadata_writer.dataframe_manipulations import (
        _add_prior_omissions,
        _add_session_number,
        _add_age_in_days,
        _patch_date_and_stage_from_pickle_file,
        _add_experience_level,
        _add_images_from_behavior,
        remove_aborted_sessions)

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .util.prior_exposure_processing import (
        get_prior_exposures_to_image_set,
        get_prior_exposures_to_session_type)

from allensdk.brain_observatory.behavior.behavior_project_cache.tables \
    .util.image_presentation_utils import (
        get_image_set)


def get_list_of_bad_probe_ids(
        lims_connection: PostgresQueryMixin,
        probes_to_skip: List[Dict[str, Any]]) -> List[int]:
    """
    Given a list of probes to skip,each of the form

    {'session': ecephys_session_id,
     'probe': probe_name}

    return a list of the ecephys_probe_ids associated with
    the bad probes.

    Parameters
    ----------
    lims_connection: PostgressQueryMixin

    probes_to_skip: List[Dict[str, Any]]
        List of dicts specifying the probes to skip (see above
        for the form each dict needs to take)

    Returns
    -------
    bad_probe_id_list: List[int]
        List of the globally unique ecephys_probe_id values
        that need to be skipped
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

    bad_probe_id_list = lims_connection.fetchall(query)
    return bad_probe_id_list


def units_table_from_ecephys_session_ids(
        lims_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will return the units table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    units_table: pd.DataFrame
        A pandas DataFrame corresponding to the units metadata table

        The columns of the dataframe are
        ================================
        unit_id -- int64 uniquely identifying this unit
        ecephys_channel_id -- int64 uniquely identifying the channel
        ecephys_probe_id -- int64 uniquely identifying the probe
        ecephys_session_id -- int64 uniquely identifying teh session
        snr -- float64
        firing_rate -- float64
        isi_violations -- float64
        presence_ratio -- float64
        amplitude_cutoff -- float64
        isolation_distance -- float64
        l_ratio -- float64
        d_prime -- float64
        nn_hit_rate -- float64
        nn_miss_rate -- float64
        silhouette_score -- float64
        max_drift -- float64
        cumulative_drift -- float64
        waveform_duration -- float64
        waveform_halfwidth -- float64
        waveform_pt_ratio -- float64
        waveform_repolarization_slope -- float64
        waveform_recovery_slope -- float64
        waveform_amplitude -- float64
        waveform_spread -- float64
        waveform_velocity_above -- float64
        waveform_velocity_below -- float64
        local_index -- int64
        probe_vertical_position -- float64
        probe_horizontal_position -- float64
        anterior_posterior_ccf_coordinate -- float64
        dorsal_ventral_ccf_coordinate -- float64
        ecephys_structure_id -- int64 uniquely identifying the structure
        ecephys_structure_acronym -- a string naming the structure
        valid_data -- a boolean indicating the validity of the channel
    """

    query = """
    SELECT
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
      ,ecephys_channels.valid_data as valid_data
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

    query += build_in_list_selector_query(
            col='ecephys_sessions.id',
            valid_list=ecephys_session_id_list,
            operator='WHERE',
            valid=True)

    if probe_ids_to_skip is not None:
        skip_clause = build_in_list_selector_query(
                        col='ecephys_probes.id',
                        valid_list=probe_ids_to_skip,
                        operator='AND',
                        valid=False)
        query += f"""{skip_clause}"""

    units_table = lims_connection.select(query)

    return units_table


def probes_table_from_ecephys_session_id_list(
        lims_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will return the probes table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    probes_table: pd.DataFrame
        A pandas DataFrame corresponding to the probes metadata table

        Columns in this dataframe are
        ==============================
        ecephys_probe_id -- int64
        ecephys_session_id -- int64
        name -- string like 'probeA', 'probeB', etc.
        sampling_rate -- float64
        lfp_sampling_rate -- float64
        phase -- float64
        has_lfp_data -- bool
        unit_count -- int64 number of units on this probe
        channel_count -- int64 number of channels on this probe
        ecephys_structure_acronyms -- a list of strings identifing all
                                      structures incident to this probe
    """

    query = """
    SELECT
      ecephys_probes.id as ecephys_probe_id
      ,ecephys_probes.ecephys_session_id
      ,ecephys_probes.name
      ,ecephys_probes.global_probe_sampling_rate as sampling_rate
      ,ecephys_probes.global_probe_lfp_sampling_rate as lfp_sampling_rate
      ,ecephys_probes.phase
      ,ecephys_probes.use_lfp_data as has_lfp_data
      ,COUNT(DISTINCT(ecephys_units.id)) AS unit_count
      ,COUNT(DISTINCT(ecephys_channels.id)) AS channel_count
      ,ARRAY_AGG(DISTINCT(structures.acronym)) AS ecephys_structure_acronyms"""

    query += """
    FROM  ecephys_probes
    JOIN ecephys_sessions
      ON ecephys_probes.ecephys_session_id = ecephys_sessions.id
    LEFT OUTER JOIN ecephys_channels
      ON ecephys_channels.ecephys_probe_id = ecephys_probes.id
    LEFT OUTER JOIN ecephys_units
      ON ecephys_units.ecephys_channel_id=ecephys_channels.id
    LEFT JOIN structures
      ON structures.id = ecephys_channels.manual_structure_id
    """

    query += build_in_list_selector_query(
            col='ecephys_sessions.id',
            valid_list=ecephys_session_id_list,
            operator='WHERE',
            valid=True)

    if probe_ids_to_skip is not None:
        skip_clause = build_in_list_selector_query(
                        col='ecephys_probes.id',
                        valid_list=probe_ids_to_skip,
                        operator='AND',
                        valid=False)
        query += f"""{skip_clause}"""

    query += """group by ecephys_probes.id"""

    probes_table = lims_connection.select(query)
    return probes_table


def channels_table_from_ecephys_session_id_list(
        lims_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will return the channels table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    channels_table: pd.DataFrame
        A pandas DataFrame corresponding to the channels metadata table

        Columns in this dataframe are
        =============================
        ecephys_channel_id -- int64
        ecephys_probe_id -- int64
        ecephys_session_id -- int64
        local_index -- int64
        probe_vertical_position -- float64
        probe_horizontal_position -- float64
        anterior_posterior_ccf_coordinate -- float64
        dorsal_ventral_ccf_coordinate -- float64
        left_right_ccf_coordinate -- float64
        ecephys_structure_acronym -- string
        unit_count -- int64 number of units on this channel
        valid_data -- a boolean indicating the validity of the channel
    """

    query = """
    SELECT
       ecephys_channels.id as ecephys_channel_id
      ,ecephys_channels.ecephys_probe_id
      ,ecephys_sessions.id AS ecephys_session_id
      ,ecephys_channels.local_index
      ,ecephys_channels.probe_vertical_position
      ,ecephys_channels.probe_horizontal_position
      ,ecephys_channels.anterior_posterior_ccf_coordinate
      ,ecephys_channels.dorsal_ventral_ccf_coordinate
      ,ecephys_channels.left_right_ccf_coordinate
      ,structures.acronym AS ecephys_structure_acronym
      ,COUNT(DISTINCT(ecephys_units.id)) AS unit_count
      ,ecephys_channels.valid_data as valid_data
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
      ON structures.id = ecephys_channels.manual_structure_id
    """

    query += build_in_list_selector_query(
            col='ecephys_sessions.id',
            valid_list=ecephys_session_id_list,
            operator='WHERE',
            valid=True)

    if probe_ids_to_skip is not None:
        skip_clause = build_in_list_selector_query(
                        col='ecephys_probes.id',
                        valid_list=probe_ids_to_skip,
                        operator='AND',
                        valid=False)
        query += f"""{skip_clause}"""

    query += """
    GROUP BY
      ecephys_channels.id,
      ecephys_sessions.id,
      structures.acronym"""

    channels_table = lims_connection.select(query)
    return channels_table


def _ecephys_summary_table_from_ecephys_session_id_list(
        lims_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int]) -> pd.DataFrame:
    """
    Perform the database query that will return the session summary table.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    Returns
    -------
    sumary_table: pd.DataFrame
        A pandas DataFrame corresponding to the session summary table

        Columns in this dataframe will be
        =================================
        ecephys_session_id -- int
        behavior_session_id -- int
        date_of_acquisition -- pd.Timestamp
        equipment_name -- string
        mouse_id -- string
        genotype -- tring
        sex -- string
        project_code -- string
        age_in_days -- int

    """
    query = """
        SELECT
          ecephys_sessions.id AS ecephys_session_id
          ,behavior_sessions.id AS behavior_session_id
          ,ecephys_sessions.date_of_acquisition
          ,equipment.name AS equipment_name
          ,ecephys_sessions.stimulus_name AS session_type
          ,donors.external_donor_name AS mouse_id
          ,donors.full_genotype AS genotype
          ,genders.name AS sex
          ,projects.code AS project_code
          ,DATE_PART('day',
            ecephys_sessions.date_of_acquisition - donors.date_of_birth)
            AS age_in_days
        """

    query += """
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
        """

    query += build_in_list_selector_query(
            col='ecephys_sessions.id',
            valid_list=ecephys_session_id_list,
            operator='WHERE',
            valid=True)

    summary_table = lims_connection.select(query)
    return summary_table


def _ecephys_counts_per_session_from_ecephys_session_id_list(
        lims_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will produce a table enumerating
    how many units, channels, and probes there are in each
    session

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    counts_table: pd.DataFrame
        A pandas DataFrame corresponding to the counts_per_session
        table

        Columns in this dataframe will be
        =================================
        ecephys_session_id -- int
        unit_count -- int (the number of units in this session)
        probe_count -- int (the number of probes in this session)
        channel_count -- int(the number of channels in this session)
    """

    query = """
    SELECT ecephys_sessions.id as ecephys_session_id,
      COUNT(DISTINCT(ecephys_units.id)) AS unit_count,
      COUNT(DISTINCT(ecephys_probes.id)) AS probe_count,
      COUNT(DISTINCT(ecephys_channels.id)) AS channel_count
    FROM ecephys_sessions
    LEFT OUTER JOIN ecephys_probes ON
      ecephys_probes.ecephys_session_id = ecephys_sessions.id
    LEFT OUTER JOIN ecephys_channels ON
      ecephys_channels.ecephys_probe_id = ecephys_probes.id
    LEFT OUTER JOIN ecephys_units ON
      ecephys_units.ecephys_channel_id = ecephys_channels.id
    """

    query += build_in_list_selector_query(
            col='ecephys_sessions.id',
            valid_list=ecephys_session_id_list,
            operator='WHERE',
            valid=True)

    if probe_ids_to_skip is not None:
        skip_clause = build_in_list_selector_query(
                        col='ecephys_probes.id',
                        valid_list=probe_ids_to_skip,
                        operator='AND',
                        valid=False)
        query += f"""{skip_clause}"""

    query += """
    GROUP BY ecephys_sessions.id"""

    counts_table = lims_connection.select(query)
    return counts_table


def _ecephys_structure_acronyms_from_ecephys_session_id_list(
        lims_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]) -> pd.DataFrame:
    """
    Perform the database query that will produce a table listing the
    acronyms of all of the structures encompassed by a given
    session.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    structure_table: pd.DataFrame
        A pandas DataFrame corresponding to the table of
        structure acronyms per session

        Columns in this dataframe will be
        =================================
        ecephys_session_id -- int
        ecephys_structure_acronyms -- a list of strings
    """
    query = """
    SELECT
      ecephys_sessions.id AS ecephys_session_id
      ,ARRAY_AGG(DISTINCT(structures.acronym)) AS ecephys_structure_acronyms
    FROM ecephys_sessions
    JOIN ecephys_probes
      ON ecephys_probes.ecephys_session_id = ecephys_sessions.id
    JOIN ecephys_channels
      ON ecephys_channels.ecephys_probe_id = ecephys_probes.id
    LEFT JOIN structures
      ON structures.id = ecephys_channels.manual_structure_id
    """

    query += build_in_list_selector_query(
            col='ecephys_sessions.id',
            valid_list=ecephys_session_id_list,
            operator='WHERE',
            valid=True)

    if probe_ids_to_skip is not None:
        skip_clause = build_in_list_selector_query(
                        col='ecephys_probes.id',
                        valid_list=probe_ids_to_skip,
                        operator='AND',
                        valid=False)
        query += f"""{skip_clause}"""

    query += """
    GROUP BY ecephys_sessions.id"""

    struct_tbl = lims_connection.select(query)
    return struct_tbl


def _behavior_session_table_from_ecephys_session_id_list(
        lims_connection: PostgresQueryMixin,
        mtrain_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int],
        exclude_sessions_after_death_date: bool = True,
        exclude_aborted_sessions: bool = True
) -> pd.DataFrame:
    """
    Given a list of ecephys_session_ids, find all of the behavior_sessions
    experienced by the same mice and return a table summarizing those
    sessions.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    mtrain_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]
        The list of ecephys_session_ids used to lookup the mice
        we are interested in following

    exclude_sessions_after_death_date
        Whether to exclude sessions that fall after death date
        in order to filter out these mistakenly entered sessions

    exclude_aborted_sessions
        Whether to exclude aborted sessions. The way that we determine if a
        session is aborted is by comparing the session duration to an
        expected duration

    Returns
    -------
    behavior_session_df: pd.DataFrame
        A dataframe summarizing the behavior sessions involving the
        mice in question.

        The columns in this dataframe are
        =================================
        mouse_id  --  str
        behavior_session_id  --  int64
        date_of_acquisition  --  datetime64[ns]
        ecephys_session_id  --  float64
        date_of_birth  --  datetime64[ns]
        genotype  --  str
        sex  --  str
        equipment_name  --  str
        foraging_id  --  str
        session_type  --  str
        image_set  --  str
        prior_exposures_to_session_type  --  int64
        prior_exposures_to_image_set  --  float64
        age_in_days  --  int64
        session_number  --  int64
    """
    behavior_session_df = behavior_sessions_from_ecephys_session_ids(
                            lims_connection=lims_connection,
                            ecephys_session_id_list=ecephys_session_id_list)

    beh_to_foraging_df = foraging_id_map_from_behavior_session_id(
        lims_engine=lims_connection,
        behavior_session_ids=behavior_session_df.behavior_session_id.tolist())

    # add foraging_id
    behavior_session_df = behavior_session_df.join(
                    beh_to_foraging_df.set_index('behavior_session_id'),
                    on='behavior_session_id',
                    how='left')

    foraging_ids = beh_to_foraging_df.foraging_id.tolist()

    # this is necessary because there are some sessions with the
    # invalid foraging_id entry 'DoC' in MTRAIN
    foraging_ids = _sanitize_uuid_list(foraging_ids)

    foraging_to_stage_df = session_stage_from_foraging_id(
            mtrain_engine=mtrain_connection,
            foraging_ids=foraging_ids)

    # add session_type
    behavior_session_df = behavior_session_df.join(
                foraging_to_stage_df.set_index('foraging_id'),
                on='foraging_id',
                how='left')

    # The date_of_acquisition and session_type stored in the LIMS
    # behavior_sessions table is untrustworthy. We will set those
    # columns to None here so that they all get patched from the
    # pickle files.

    behavior_session_df.date_of_acquisition = None
    behavior_session_df.session_type = None

    behavior_session_df = _patch_date_and_stage_from_pickle_file(
                             lims_connection=lims_connection,
                             behavior_df=behavior_session_df)

    if exclude_sessions_after_death_date:
        # filter out any sessions which were mistakenly entered that fall
        # after mouse death date
        behavior_session_df = behavior_session_df[
            behavior_session_df['date_of_acquisition'] <=
            behavior_session_df.merge(get_death_date_for_mouse_ids(
                lims_connections=lims_connection,
                mouse_ids_list=behavior_session_df['mouse_id'].tolist()),
                on='mouse_id')['death_on']
        ]

    if exclude_aborted_sessions:
        behavior_session_df = remove_aborted_sessions(
            lims_connection=lims_connection,
            behavior_df=behavior_session_df
        )

    behavior_session_df['image_set'] = get_image_set(
            df=behavior_session_df)

    behavior_session_df['prior_exposures_to_session_type'] = \
        get_prior_exposures_to_session_type(
            df=behavior_session_df)

    behavior_session_df['prior_exposures_to_image_set'] = \
        get_prior_exposures_to_image_set(
            df=behavior_session_df)

    behavior_session_df = _add_age_in_days(
        df=behavior_session_df)

    behavior_session_df = _add_session_number(
        sessions_df=behavior_session_df,
        index_col="behavior_session_id")

    return behavior_session_df


def session_tables_from_ecephys_session_id_list(
        lims_connection: PostgresQueryMixin,
        mtrain_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int],
        probe_ids_to_skip: Optional[List[int]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform the database query to generate the ecephys_session_table

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    mtrain_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]
        The list of ecephys_sessions.id values of the
        ecephys sessions for which to construct the units table

    probe_ids_to_skip: Optional[List[int]]
        The IDs of probes not being released

    Returns
    -------
    session_table: pd.DataFrame
        A pandas DataFrame corresponding to the session table

        This dataframe has columns
        ==========================
        ecephys_session_id -- int64
        behavior_session_id -- int64
        date_of_acquisition -- pd.Timestamp
        equipment_name -- str
        session_type -- str
        mouse_id -- int64
        genotype -- str
        sex -- str
        project_code -- str
        age_in_days -- float64
        unit_count -- int64
        probe_count -- int64
        channel_count -- int64
        ecephys_structure_acronyms -- list of strings
        image_set -- str
        prior_exposures_to_image_set -- float64
        session_number -- int64
        prior_exposures_to_omissions -- int64
        experience_level -- str

    behavior_session-table: pd.DataFrame

        This dataframe has columns
        ==========================
        behavior_session_id -- int64
        equipment_name -- str
        genotype -- str
        mouse_id -- int64
        sex -- str
        age_in_days -- int64
        session_number -- int64
        prior_exposures_to_session_type -- int64
        prior_exposures_to_image_set -- float64
        prior_exposures_to_omissions -- int64
        ecephys_session_id -- float64
        date_of_acquisition -- str
        session_type -- str
        image_set -- str
    """

    beh_table = _behavior_session_table_from_ecephys_session_id_list(
            lims_connection=lims_connection,
            mtrain_connection=mtrain_connection,
            ecephys_session_id_list=ecephys_session_id_list)

    summary_tbl = _ecephys_summary_table_from_ecephys_session_id_list(
                        lims_connection=lims_connection,
                        ecephys_session_id_list=ecephys_session_id_list)

    # patch date_of_acquisition and session_type from beh_table,
    # which read them directly from the pickle file
    summary_tbl = patch_df_from_other(
                    target_df=summary_tbl,
                    source_df=beh_table,
                    index_column='behavior_session_id',
                    columns_to_patch=['date_of_acquisition',
                                      'session_type'])

    ct_tbl = _ecephys_counts_per_session_from_ecephys_session_id_list(
                        lims_connection=lims_connection,
                        ecephys_session_id_list=ecephys_session_id_list,
                        probe_ids_to_skip=probe_ids_to_skip)

    summary_tbl = summary_tbl.join(
                        ct_tbl.set_index("ecephys_session_id"),
                        on="ecephys_session_id",
                        how='left')

    struct_tbl = _ecephys_structure_acronyms_from_ecephys_session_id_list(
                        lims_connection=lims_connection,
                        ecephys_session_id_list=ecephys_session_id_list,
                        probe_ids_to_skip=probe_ids_to_skip)

    summary_tbl = summary_tbl.join(
                     struct_tbl.set_index("ecephys_session_id"),
                     on="ecephys_session_id",
                     how='left')

    summary_tbl = _add_images_from_behavior(
            ecephys_table=summary_tbl,
            behavior_table=beh_table)

    sessions_table = _add_session_number(
                            sessions_df=summary_tbl,
                            index_col="ecephys_session_id")
    sessions_table = _add_experience_level(
                            sessions_df=sessions_table)

    omission_results = _add_prior_omissions(
                behavior_sessions_df=beh_table,
                ecephys_sessions_df=sessions_table)

    beh_table = omission_results['behavior']
    sessions_table = omission_results['ecephys']

    beh_table = beh_table[
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

    return sessions_table, beh_table
