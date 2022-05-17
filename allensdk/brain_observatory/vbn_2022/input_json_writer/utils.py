from typing import List, Optional, Dict, Union

import pandas as pd
import numpy as np
import numbers

from allensdk.internal.api.queries.wkf_lims_queries import (
    wkf_path_from_attachable)

from allensdk.internal.api.queries.equipment_lims_queries import (
    experiment_configs_from_equipment_id_and_type)

from allensdk.internal.api import PostgresQueryMixin
from allensdk import OneResultExpectedError
from allensdk.internal.api.queries.utils import (
    build_in_list_selector_query)

from allensdk.brain_observatory.behavior.data_objects.\
    metadata.subject_metadata.reporter_line import ReporterLine

from allensdk.brain_observatory.behavior.data_objects.\
    metadata.subject_metadata.driver_line import DriverLine

from allensdk.brain_observatory.vbn_2022.metadata_writer.lims_queries import (
    _ecephys_summary_table_from_ecephys_session_id_list,
    probes_table_from_ecephys_session_id_list,
    channels_table_from_ecephys_session_id_list,
    units_table_from_ecephys_session_id_list,
    get_list_of_bad_probe_ids)

from allensdk.brain_observatory.vbn_2022.metadata_writer.\
    dataframe_manipulations import (
        _add_age_in_days)

from allensdk.core.auth_config import (
    LIMS_DB_CREDENTIAL_MAP)

from allensdk.internal.api import db_connection_creator


class NwbConfigErrorLog(object):

    def __init__(self):
        self._messages = dict()

    def log(self, ecephys_session_id: Union[int, str], msg: str):
        ecephys_session_id = int(ecephys_session_id)
        if ecephys_session_id not in self._messages:
            self._messages[ecephys_session_id] = []
        self._messages[ecephys_session_id].append(msg)

    def write(self) -> str:
        k_list = list(self._messages.keys())
        k_list.sort()
        msg = ""
        for k in k_list:
            msg += f"ecephys_session: {k}\n"
            for m in self._messages[k]:
                msg += f"    {m}\n"
        return msg


def vbn_nwb_config_from_ecephys_session_id_list(
        ecephys_session_id_list: List[int],
        probes_to_skip: Optional[List[dict]]) -> List[dict]:

    error_log = NwbConfigErrorLog()

    lims_connection = db_connection_creator(
            fallback_credentials=LIMS_DB_CREDENTIAL_MAP)

    if probes_to_skip is not None:
        probe_ids_to_skip = get_list_of_bad_probe_ids(
                lims_connection=lims_connection,
                probes_to_skip=probes_to_skip)
    else:
        probe_ids_to_skip=None

    session_list = session_input_from_ecephys_session_id_list(
            ecephys_session_id_list=ecephys_session_id_list,
            lims_connection=lims_connection,
            error_log=error_log)

    for session in session_list:
        session_id = session['ecephys_session_id']

        probe_list = probe_input_from_ecephys_session_id(
                        ecephys_session_id=session_id,
                        probe_ids_to_skip=probe_ids_to_skip,
                        lims_connection=lims_connection,
                        error_log=error_log)

        session['probes'] = probe_list

        channel_input = channel_input_from_ecephys_session_id(
                            ecephys_session_id=session_id,
                            probe_ids_to_skip=probe_ids_to_skip,
                            lims_connection=lims_connection,
                            error_log=error_log)

        bad_probe_list = []
        for idx, probe in enumerate(session['probes']):
            probe_id = probe['id']
            if probe_id in channel_input:
                channels = channel_input[probe_id]
                probe['channels'] = channels
            else:
                bad_probe_list.append(idx)
                msg = (f"could not find channels for probe {probe_id}; "
                       "not listing in the input.json")
                error_log.log(ecephys_session_id=session_id,
                              msg=msg)

        # remove probes that have no channels
        bad_probe_list.reverse()
        for idx in bad_probe_list:
            this_probe = session['probes'].pop(idx)
            assert 'channels' not in this_probe

        unit_input = unit_input_from_ecephys_session_id(
                        ecephys_session_id=session_id,
                        probe_ids_to_skip=probe_ids_to_skip,
                        lims_connection=lims_connection,
                        error_log=error_log)

        for probe in session['probes']:
            probe_id = probe['id']
            if probe_id in unit_input:
                units = unit_input[probe_id]
                probe['units'] = units
            else:
                msg = f"could not find units for probe {probe_id}"
                error_log.log(ecephys_session_id=session_id,
                              msg=msg)

    print(error_log.write())
    return session_list


def session_input_from_ecephys_session_id_list(
        ecephys_session_id_list: List[int],
        lims_connection: PostgresQueryMixin,
        error_log: NwbConfigErrorLog) -> List[dict]:
    """
    Take list of session IDs; return a list of dicts, each dict
    representing the input json needed for a session
    """

    session_table = _ecephys_summary_table_from_ecephys_session_id_list(
            lims_connection=lims_connection,
            ecephys_session_id_list=ecephys_session_id_list)

    session_table.rename(
            columns={'equipment_name': 'rig_name',
                     'mouse_id': 'external_specimen_name',
                     'genotype': 'full_genotype'},
            inplace=True)

    session_table.external_specimen_name = \
        session_table.external_specimen_name.astype(int)

    session_table = _add_age_in_days(
                df=session_table,
                index_column='ecephys_session_id')

    session_table.age_in_days = session_table.age_in_days.apply(
                         lambda x: f'P{int(x):d}')

    session_table.rename(
            columns={'equipment_name': 'rig_name',
                     'mouse_id': 'external_specimen_name',
                     'genotype': 'full_genotype',
                     'age_in_days': 'age'},
            inplace=True)

    session_table.drop(
        labels=['session_type', 'project_code'],
        axis='columns',
        inplace=True)

    session_table = session_table.set_index(
                        'ecephys_session_id')

    session_table = session_table.to_dict(orient='index')

    # get lookup tables mapping ecephys_session_id to the
    # ecephys_analysis_run_ids for the optotagging and
    # stimulus table files
    optotagging_run_lookup = _analysis_run_from_session_id(
            lims_connection=lims_connection,
            ecephys_session_id_list=ecephys_session_id_list,
            strategy_class='EcephysOptotaggingTableStrategy')

    stim_table_run_lookup = _analysis_run_from_session_id(
            lims_connection=lims_connection,
            ecephys_session_id_list=ecephys_session_id_list,
            strategy_class='VbnCreateStimTableStrategy')

    # well known files that are linked to the ecephys sesssion
    # (as opposed to an ecephys analysis run)
    input_from_wkf_session = [
            ('behavior_stimulus_file', 'StimulusPickle'),
            ('mapping_stimulus_file', 'MappingPickle'),
            ('replay_stimulus_file', 'EcephysReplayStimulus'),
            ('raw_eye_tracking_video_meta_data',
             'RawEyeTrackingVideoMetadata'),
            ('eye_dlc_file', 'EyeDlcOutputFile'),
            ('face_dlc_file', 'FaceDlcOutputFile'),
            ('side_dlc_file', 'SideDlcOutputFile'),
            ('eye_tracking_filepath', 'EyeTracking Ellipses'),
            ('sync_file', 'EcephysRigSync')]

    wkf_types_to_query_session = [f"'{el[1]}'"
                                  for el in input_from_wkf_session]

    result = []
    for session_id in ecephys_session_id_list:
        session_id = int(session_id)
        if session_id not in session_table:
            error_log.log(ecephys_session_id=session_id,
                          msg="No session data was found at all; skipping")
            continue

        data = session_table[session_id]
        data['ecephys_session_id'] = session_id

        wkf_path_lookup = wkf_path_from_attachable(
                            lims_connection=lims_connection,
                            wkf_type_name=wkf_types_to_query_session,
                            attachable_type='EcephysSession',
                            attachable_id=session_id)

        for key_pair in input_from_wkf_session:
            this_path = wkf_path_lookup.get(key_pair[1], None)
            if this_path is None:
                msg = (f"Could not find {key_pair[1]} "
                       f"for ecephys_session {session_id}")
                error_log.log(ecephys_session_id=session_id,
                              msg=msg)
            data[key_pair[0]] = this_path

        # get stimulus_table
        stim_path_lookup = wkf_path_from_attachable(
                    lims_connection=lims_connection,
                    wkf_type_name=["'EcephysStimulusTable'", ],
                    attachable_type="EcephysAnalysisRun",
                    attachable_id=stim_table_run_lookup[session_id])

        data['stim_table_file'] = stim_path_lookup['EcephysStimulusTable']

        # get optotagging_table
        optotagging_path_lookup = wkf_path_from_attachable(
                    lims_connection=lims_connection,
                    wkf_type_name=["'EcephysOptotaggingTable'", ],
                    attachable_type="EcephysAnalysisRun",
                    attachable_id=optotagging_run_lookup[session_id])

        data['optotagging_table_path'] = optotagging_path_lookup[
                        "EcephysOptotaggingTable"]

        driver_line = DriverLine.from_lims(
                        lims_db=lims_connection,
                        behavior_session_id=data['behavior_session_id'],
                        allow_none=True).value

        if driver_line is not None:
            data['driver_line'] = driver_line
        else:
            data['driver_line'] = []

        reporter_line = ReporterLine.from_lims(
                            lims_db=lims_connection,
                            behavior_session_id=data['behavior_session_id'],
                            allow_none=True).value

        if reporter_line is not None:
            if isinstance(reporter_line, list):
                data['reporter_line'] = reporter_line
            else:
                data['reporter_line'] = [reporter_line, ]
        else:
            data['reporter_line'] = []

        eye_geometry = eye_tracking_geometry_from_equipment_id(
                equipment_id=data.pop('equipment_id'),
                date_of_acquisition=data['date_of_acquisition'],
                lims_connection=lims_connection)

        eye_geometry['equipment'] = data['rig_name']

        data['eye_tracking_rig_geometry'] = eye_geometry

        for k in ('date_of_acquisition',
                  'date_of_birth'):
            data[k] = str(data[k])

        result.append(data)

    return result


def probe_input_from_ecephys_session_id(
        ecephys_session_id: int,
        probe_ids_to_skip: Optional[List[int]],
        lims_connection: PostgresQueryMixin,
        error_log: NwbConfigErrorLog) -> List[dict]:

    probes_table = probes_table_from_ecephys_session_id_list(
                        lims_connection=lims_connection,
                        ecephys_session_id_list=[ecephys_session_id, ],
                        probe_ids_to_skip=probe_ids_to_skip)

    probes_table = probes_table.set_index('ecephys_probe_id')

    probes_table.drop(
        labels=['ecephys_session_id',
                'phase',
                'has_lfp_data',
                'unit_count',
                'channel_count',
                'ecephys_structure_acronyms'],
        axis='columns',
        inplace=True)

    probes_table = probes_table.to_dict(orient='index')

    input_from_wkf_probe = [
        ('inverse_whitening_matrix_path', 'EcephysSortedWhiteningMatInv'),
        ('mean_waveforms_path', 'EcephysSortedMeanWaveforms'),
        ('spike_amplitudes_path', 'EcephysSortedAmplitudes'),
        ('spike_clusters_file', 'EcephysSortedSpikeClusters'),
        ('spike_templates_path', 'EcephysSortedSpikeTemplates'),
        ('templates_path', 'EcephysSortedTemplates')]

    wkf_to_query = [f"'{el[1]}'"
                    for el in input_from_wkf_probe]

    results = []
    probe_id_list = list(probes_table.keys())
    probe_id_list.sort()

    for probe_id in probe_id_list:
        data = probes_table[probe_id]
        data['csd_path'] = None
        data['lfp'] = None
        data['id'] = probe_id

        wkf_path_lookup = wkf_path_from_attachable(
                            lims_connection=lims_connection,
                            wkf_type_name=wkf_to_query,
                            attachable_type='EcephysProbe',
                            attachable_id=probe_id)
        for key_pair in input_from_wkf_probe:
            data[key_pair[0]] = wkf_path_lookup.get(key_pair[1], None)

        results.append(_nan_to_none(data))

    _add_spike_times_path(
        data=results,
        ecephys_session_id=ecephys_session_id,
        lims_connection=lims_connection,
        error_log=error_log)

    return results


def channel_input_from_ecephys_session_id(
        ecephys_session_id: int,
        probe_ids_to_skip: Optional[List[int]],
        lims_connection: PostgresQueryMixin,
        error_log: NwbConfigErrorLog) -> Dict[int, list]:
    """
    Returns a dict mapping probe_id to the list of channel specifications
    """

    raw_channels_table = channels_table_from_ecephys_session_id_list(
                                ecephys_session_id_list=[ecephys_session_id, ],
                                probe_ids_to_skip=probe_ids_to_skip,
                                lims_connection=lims_connection)

    raw_channels_table.rename(
            columns={'ecephys_channel_id': 'id',
                     'ecephys_probe_id': 'probe_id',
                     'ecephys_structure_acronym': 'manual_structure_acronym',
                     'ecephys_structure_id': 'manual_structure_id'},
            inplace=True)

    raw_channels_table = raw_channels_table[[
                              'id',
                              'probe_id',
                              'local_index',
                              'manual_structure_id',
                              'manual_structure_acronym',
                              'anterior_posterior_ccf_coordinate',
                              'dorsal_ventral_ccf_coordinate',
                              'left_right_ccf_coordinate',
                              'probe_horizontal_position',
                              'probe_vertical_position',
                              'valid_data']]
    raw_channels_table = raw_channels_table.set_index('id')
    raw_channels_table = raw_channels_table.to_dict(orient='index')
    output_dict = dict()
    for channel_id in raw_channels_table.keys():
        this_channel = raw_channels_table[channel_id]
        probe_id = this_channel['probe_id']
        if probe_id not in output_dict:
            output_dict[probe_id] = []
        this_channel['id'] = channel_id
        output_dict[probe_id].append(_nan_to_none(this_channel))
    return output_dict


def unit_input_from_ecephys_session_id(
        ecephys_session_id: int,
        probe_ids_to_skip: Optional[List[int]],
        lims_connection: PostgresQueryMixin,
        error_log: NwbConfigErrorLog) -> Dict[int, list]:
    """
    Returns a dict mapping probe_id to the list of unit specifications
    """
    raw_unit_table = units_table_from_ecephys_session_id_list(
                        ecephys_session_id_list=[ecephys_session_id, ],
                        probe_ids_to_skip=probe_ids_to_skip,
                        lims_connection=lims_connection)

    raw_unit_table.rename(
        columns={'unit_id': 'id',
                 'waveform_pt_ratio': 'PT_ratio',
                 'waveform_amplitude': 'amplitude',
                 'ecephys_channel_id': 'peak_channel_id',
                 'waveform_velocity_above': 'velocity_above',
                 'waveform_velocity_below': 'velocity_below',
                 'waveform_repolarization_slope': 'repolarization_slope',
                 'waveform_recovery_slope': 'recovery_slope',
                 'waveform_spread': 'spread'},
        inplace=True)

    if len(raw_unit_table) == 0:
        msg = f"could not find units for session {ecephys_session_id}"
        error_log.log(ecephys_session_id=ecephys_session_id,
                      msg=msg)
        return dict()

    raw_unit_table.drop(
        labels=['ecephys_session_id',
                'probe_vertical_position',
                'probe_horizontal_position',
                'anterior_posterior_ccf_coordinate',
                'dorsal_ventral_ccf_coordinate',
                'ecephys_structure_id',
                'ecephys_structure_acronym',
                'valid_data'],
        axis='columns',
        inplace=True)

    raw_unit_table = raw_unit_table.set_index('id')
    raw_unit_table = raw_unit_table.to_dict(orient='index')
    output_dict = dict()
    for unit_id in raw_unit_table.keys():
        this_unit = raw_unit_table[unit_id]
        this_unit['id'] = unit_id
        probe_id = this_unit.pop('ecephys_probe_id')
        if probe_id not in output_dict:
            output_dict[probe_id] = []
        output_dict[probe_id].append(_nan_to_none(this_unit))
    return output_dict


def eye_tracking_geometry_from_equipment_id(
        equipment_id: int,
        date_of_acquisition: pd.Timestamp,
        lims_connection: PostgresQueryMixin) -> dict:

    """
    Return eye_tracking_rig_geometry given a specified
    equipment_id and date_of_acquisition

    Parameters
    ----------
    equipment_id: int

    date_of_acqisition: pd.Timestamp

    lims_connection: PostgresQueryMixin

    Returns
    --------
    eye_geometry: dict
        The eye_tracking_geometry dict to be written to
        the input.json
    """
    raw_eye_geometry = _raw_eye_tracking_geometry_from_equipment_id(
            equipment_id=equipment_id,
            date_of_acquisition=date_of_acquisition,
            lims_connection=lims_connection)

    eye_geometry = dict()
    eye_geometry['led_position'] = [
            raw_eye_geometry['led position']['center_x_mm'],
            raw_eye_geometry['led position']['center_y_mm'],
            raw_eye_geometry['led position']['center_z_mm']]

    eye_geometry['monitor_position_mm'] = [
            raw_eye_geometry['screen position']['center_x_mm'],
            raw_eye_geometry['screen position']['center_y_mm'],
            raw_eye_geometry['screen position']['center_z_mm']]

    eye_geometry['monitor_rotation_deg'] = [
            raw_eye_geometry['screen position']['rotation_x_deg'],
            raw_eye_geometry['screen position']['rotation_y_deg'],
            raw_eye_geometry['screen position']['rotation_z_deg']]

    eye_geometry['camera_position_mm'] = [
            raw_eye_geometry['eye camera position']['center_x_mm'],
            raw_eye_geometry['eye camera position']['center_y_mm'],
            raw_eye_geometry['eye camera position']['center_z_mm']]

    eye_geometry['camera_rotation_deg'] = [
            raw_eye_geometry['eye camera position']['rotation_x_deg'],
            raw_eye_geometry['eye camera position']['rotation_y_deg'],
            raw_eye_geometry['eye camera position']['rotation_z_deg']]

    return eye_geometry


def _raw_eye_tracking_geometry_from_equipment_id(
        equipment_id: int,
        date_of_acquisition: pd.Timestamp,
        lims_connection: PostgresQueryMixin) -> dict:
    """
    Return eye_tracking_rig_geometry given a specified
    equipment_id and date_of_acquisition

    Parameters
    ----------
    equipment_id: int

    date_of_acquisition: pd.Timestamp

    lims_connection: PostgresQueryMixin

    Returns
    -------
    config: dict
        A dict listing the configuration of the
        eye tracking rig geometry

    Notes
    -----
    Will return the configuration with the latest
    active_date that is before date_of_acquisition.
    """
    config = dict()
    for name in ('led position', 'behavior camera position',
                 'eye camera position', 'screen position'):
        this_df = experiment_configs_from_equipment_id_and_type(
                        equipment_id=equipment_id,
                        config_type=name,
                        lims_connection=lims_connection)
        this_df = this_df.loc[this_df.active_date <= date_of_acquisition]
        this_df = this_df.iloc[this_df.active_date.idxmax()]
        this_config = dict()
        this_config['center_x_mm'] = this_df.center_x_mm
        this_config['center_y_mm'] = this_df.center_y_mm
        this_config['center_z_mm'] = this_df.center_z_mm
        this_config['rotation_x_deg'] = this_df.rotation_x_deg
        this_config['rotation_y_deg'] = this_df.rotation_y_deg
        this_config['rotation_z_deg'] = this_df.rotation_z_deg
        config[name] = this_config

    return config


def _analysis_run_from_session_id(
        lims_connection: PostgresQueryMixin,
        ecephys_session_id_list: List[int],
        strategy_class: str) -> Dict[int, int]:
    """
    Get a dict mapping ecephys_session_id to ecephys_analysis_runs.id
    for a specific job strategy class ('VbnCreateStimTableStrategy',
    'EcephysOptotaggingTableStrategy', etc.). Will only select the
    instance of the run marked as 'current' in the LIMS database.

    Parameters
    ----------
    lims_connection: PostgresQueryMixin

    ecephys_session_id_list: List[int]

    strategy_class: str

    Returns
    -------
    analysis_run_map: Dict[int, int]
        A dict mapping ecephys_session_id to the
        ecephys_analysis_runs.id associated with the specified
        job strategy. Only returns the row marked as 'current'

    Notes
    -----
    Raises a OneResultExpectedError if more than one run_id is returned
    for the same session_id.
    """

    query = """
    SELECT
      ecephys_session_id
      ,id as ecephys_analysis_run_id
    FROM
      ecephys_analysis_runs
    WHERE
        ecephys_analysis_runs.current
    """

    query += build_in_list_selector_query(
                col="ecephys_analysis_runs.ecephys_session_id",
                valid_list=ecephys_session_id_list,
                operator="AND",
                valid=True)

    query += build_in_list_selector_query(
                col="ecephys_analysis_runs.job_strategy_class",
                valid_list=[f"'{strategy_class}'"],
                operator="AND",
                valid=True)

    query_result = lims_connection.select(query)
    analysis_run_map = dict()
    msg = ""
    for session_id, run_id in zip(query_result.ecephys_session_id,
                                  query_result.ecephys_analysis_run_id):

        if session_id in analysis_run_map:
            msg += ("More than one analysis run returned for "
                    f"ecephys_session_id={session_id}\n")
        analysis_run_map[session_id] = run_id

    if len(msg) > 0:
        raise OneResultExpectedError(msg)

    return analysis_run_map


def _add_spike_times_path(
        data: List[dict],
        ecephys_session_id: int,
        lims_connection: PostgresQueryMixin,
        error_log: NwbConfigErrorLog) -> List[dict]:
    """
    Add the 'spike_times_path' entry to a list of probe specifications.

    Parameters
    ----------
    data: List[dict]
        The list of probe specifications to be modified

    ecephys_session_id:int

    lims_connection: PostgresQueryMixin

    Returns
    -------
    data: List[dict]
        Same as input with 'spike_times_path' added.

    Notes
    -----
    Will alter data in place
    """

    probe_id_list = [this_probe['id'] for this_probe in data]

    timestamp_run_lookup = _analysis_run_from_session_id(
                        lims_connection=lims_connection,
                        ecephys_session_id_list=[ecephys_session_id, ],
                        strategy_class='EcephysAlignTimestampsStrategy')

    # get mapping from probe_id to
    # ecephys_analysis_run_probes.id
    query = """
    SELECT
      ecephys_analysis_run_probes.ecephys_probe_id as probe_id
      ,ecephys_analysis_run_probes.id as ecephys_analysis_run_probe_id
    FROM
      ecephys_analysis_run_probes
    """

    query += build_in_list_selector_query(
                col="ecephys_analysis_run_probes.ecephys_probe_id",
                valid_list=probe_id_list,
                operator="WHERE",
                valid=True)

    query += build_in_list_selector_query(
                col="ecephys_analysis_run_probes.ecephys_analysis_run_id",
                valid_list=[
                    timestamp_run_lookup[ecephys_session_id], ],
                operator="AND",
                valid=True)

    query_result = lims_connection.select(query)
    probe_run_lookup = dict()
    for p_id, r_id in zip(query_result.probe_id,
                          query_result.ecephys_analysis_run_probe_id):
        probe_run_lookup[int(p_id)] = int(r_id)

    for this_probe in data:
        probe_id = this_probe['id']
        if probe_id in probe_run_lookup:
            timestamp_lookup = wkf_path_from_attachable(
                            lims_connection=lims_connection,
                            wkf_type_name=[
                                "'EcephysAlignedEventTimestamps'", ],
                            attachable_type='EcephysAnalysisRunProbe',
                            attachable_id=probe_run_lookup[probe_id])

            this_probe['spike_times_path'] = \
                timestamp_lookup['EcephysAlignedEventTimestamps']
        else:
            msg = ("could not find EcephysAlignedEventTimestamps for "
                   f"probe {probe_id}")
            error_log.log(ecephys_session_id=ecephys_session_id,
                          msg=msg)

            this_probe['spike_times_path'] = None

    return data


def _nan_to_none(input_dict: dict) -> dict:
    for k in input_dict:
        val = input_dict[k]
        if isinstance(val, numbers.Number):
            if np.isnan(val):
                input_dict[k] = None
    return input_dict
