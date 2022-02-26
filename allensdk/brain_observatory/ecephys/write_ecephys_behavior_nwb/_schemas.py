from argschema import ArgSchema
from argschema.fields import (LogLevel, String, Int, Nested, List, InputFile)
import marshmallow as mm
import pandas as pd
import numpy as np

from argschema.fields import (
    LogLevel,
    Dict,
    String,
    Int,
    DateTime,
    Nested,
    Boolean,
    Float,
)

from allensdk.brain_observatory.argschema_utilities import (
    check_read_access, check_write_access_overwrite, RaisingSchema)

class Channel(RaisingSchema):

    @mm.pre_load
    def set_field_defaults(self, data, **kwargs):
        if data.get("filtering") is None:
            data["filtering"] = ("AP band: 500 Hz high-pass; "
                                 "LFP band: 1000 Hz low-pass")
        if data.get("manual_structure_acronym") is None:
            data["manual_structure_acronym"] = ""
        return data

    id = Int(required=True)
    probe_id = Int(required=True)
    valid_data = Boolean(required=True)
    local_index = Int(required=True)
    probe_vertical_position = Int(required=True)
    probe_horizontal_position = Int(required=True)
    manual_structure_id = Int(required=True, allow_none=True)
    manual_structure_acronym = String(required=True)
    anterior_posterior_ccf_coordinate = Float(allow_none=True)
    dorsal_ventral_ccf_coordinate = Float(allow_none=True)
    left_right_ccf_coordinate = Float(allow_none=True)
    impedence = Float(required=False, allow_none=True, default=None)
    filtering = String(required=False)

    @mm.post_load
    def set_impedence_default(self, data, **kwargs):
        # This must be a post_load operation as np.nan is not a valid
        # JSON format 'float' type for the Marshmallow `Float` field
        # (so validation fails if this is set at pre_load)
        if data.get("impedence") is None:
            data["impedence"] = np.nan
        return data

class Unit(RaisingSchema):
    id = Int(required=True)
    peak_channel_id = Int(required=True)
    local_index = Int(
        required=True,
        help="within-probe index of this unit.",
    )
    cluster_id = Int(
        required=True,
        help="within-probe identifier of this unit",
    )
    quality = String(required=True)
    firing_rate = Float(required=True)
    snr = Float(required=True, allow_none=True)
    isi_violations = Float(required=True)
    presence_ratio = Float(required=True)
    amplitude_cutoff = Float(required=True)
    isolation_distance = Float(required=True, allow_none=True)
    l_ratio = Float(required=True, allow_none=True)
    d_prime = Float(required=True, allow_none=True)
    nn_hit_rate = Float(required=True, allow_none=True)
    nn_miss_rate = Float(required=True, allow_none=True)
    max_drift = Float(required=True, allow_none=True)
    cumulative_drift = Float(required=True, allow_none=True)
    silhouette_score = Float(required=True, allow_none=True)
    waveform_duration = Float(required=True, allow_none=True)
    waveform_halfwidth = Float(required=True, allow_none=True)
    PT_ratio = Float(required=True, allow_none=True)
    repolarization_slope = Float(required=True, allow_none=True)
    recovery_slope = Float(required=True, allow_none=True)
    amplitude = Float(required=True, allow_none=True)
    spread = Float(required=True, allow_none=True)
    velocity_above = Float(required=True, allow_none=True)
    velocity_below = Float(required=True, allow_none=True)

class Lfp(RaisingSchema):
    input_data_path = String(required=True, validate=check_read_access)
    input_timestamps_path = String(required=True, validate=check_read_access)
    input_channels_path = String(required=True, validate=check_read_access)
    output_path = String(required=True)


class Probe(RaisingSchema):
    id = Int(required=True)
    name = String(required=True)
    spike_times_path = String(required=True, validate=check_read_access)
    spike_clusters_file = String(required=True, validate=check_read_access)
    mean_waveforms_path = String(required=True, validate=check_read_access)
    channels = Nested(Channel, many=True, required=True)
    units = Nested(Unit, many=True, required=True)
    lfp = Nested(Lfp, many=False, required=False, allow_none=True)
    csd_path = String(required=False,
                      validate=check_read_access,
                      allow_none=True,
                      help="""path to h5 file containing calculated current
                              source density""")
    sampling_rate = Float(
        default=30000.0,
        help="""sampling rate (Hz, master clock) at which raw data were
                acquired on this probe""")
    lfp_sampling_rate = Float(
        default=2500.0,
        allow_none=True,
        help="""sampling rate of LFP data on this probe""")
    temporal_subsampling_factor = Float(
        default=2.0,
        allow_none=True,
        help="""subsampling factor applied to lfp data for
                this probe (across time)""")
    spike_amplitudes_path = String(
        validate=check_read_access,
        help="""path to npy file containing scale factor applied to the
                kilosort template used to extract each spike"""
    )
    spike_templates_path = String(
        validate=check_read_access,
        help="""path to file associating each spike with a kilosort template"""
    )
    templates_path = String(
        validate=check_read_access,
        help="""path to file containing an (nTemplates)x(nSamples)x(nUnits)
                array of kilosort templates"""
    )
    inverse_whitening_matrix_path = String(
        validate=check_read_access,
        help="""Kilosort templates are whitened. In order to use them for
                scaling spike amplitudes to volts, we need to remove
                the whitening"""
    )
    amplitude_scale_factor = Float(
        default=0.195e-6,
        help="""amplitude scale factor converting raw amplitudes to Volts.
                Default converts from bits -> uV -> V"""
    )


class BehaviorSessionData(RaisingSchema):
    ecephys_session_id = Int(required=True,
                              description=("Unique identifier for the "
                                           "ecephys session to write into "
                                           "NWB format"))

    foraging_id = String(required=True,
                         description=("The foraging_id for the behavior "
                                      "session"))
    driver_line = List(String,
                       required=True,
                       description='Genetic driver line(s) of subject')
    reporter_line = List(String,
                         required=True,
                         description='Genetic reporter line(s) of subject')
    full_genotype = String(required=True,
                           description='Full genotype of subject')
    rig_name = String(required=True,
                      description=("Name of experimental rig used for "
                                   "the behavior session"))

    sex = String(required=True, description="Subject sex")
    age = String(required=True, description="Subject age")

    date_of_acquisition = String(required=True,
                                 description=("Date of acquisition of "
                                              "behavior session, in string "
                                              "format"))
    external_specimen_name = Int(required=True,
                                 description='LabTracks ID of the subject')
    behavior_pkl_path = InputFile(
        required=True,
        validate=check_read_access,
        description=("Path of behavior_stimulus "
                     "camstim *.pkl file")
    )

    replay_pkl_path = InputFile(
        required=True,
        validate=check_read_access,
        description=("Path of replay stimulus "
                     "camstim *.pkl file")
    )

    mapping_pkl_path = InputFile(
        required=True,
        validate=check_read_access,
        description=("Path of mapping stimulus "
                     "camstim *.pkl file")
    )

    sync_file = InputFile(
        required=True,
        help="path to h5 file containing synchronization information",
    )

    stim_table_file = InputFile(
        required=True,
        help="path to csv file containing stimulus information",
    )

    running_speed_path = InputFile(
        required=True,
        help="path to running speed file",
    )



    date_of_birth = String(required=True, description="Subject date of birth")
    # sex = String(required=True, description="Subject sex")
    # age = String(required=True, description="Subject age")
    # stimulus_name = String(required=True,
    #                        description=("Name of stimulus presented during "
    #                                     "behavior session"))

    # @mm.pre_load
    # def set_stimulus_name(self, data, **kwargs):
    #     if data.get("stimulus_name") is None:
    #         pkl = pd.read_pickle(data["behavior_stimulus_file"])
    #         try:
    #             stimulus_name = pkl["items"]["behavior"]["cl_params"]["stage"]
    #         except KeyError:
    #             raise mm.ValidationError(
    #                 f"Could not obtain stimulus_name/stage information from "
    #                 f"the *.pkl file ({data['behavior_stimulus_file']}) "
    #                 f"for the behavior session to save as NWB! The "
    #                 f"following series of nested keys did not work: "
    #                 f"['items']['behavior']['cl_params']['stage']"
    #             )
    #         data["stimulus_name"] = stimulus_name
    #     return data


class InputSchema(ArgSchema):
    class Meta:
        unknown = mm.RAISE
    log_level = LogLevel(default='INFO',
                         description='Logging level of the module')
    session_data = Nested(BehaviorSessionData,
                          required=True,
                          description='Data pertaining to a behavior session')
    output_path = String(required=True,
                         validate=check_write_access_overwrite,
                         description='Path of output.json to be written')

    probes = Nested(
        Probe,
        many=True,
        required=True,
        help="records of the individual probes used for this experiment",
    )



class OutputSchema(RaisingSchema):
    input_parameters = Nested(InputSchema)
    output_path = String(required=True,
                         validate=check_write_access_overwrite,
                         description='Path of output.json to be written')
