import marshmallow as mm
import numpy as np

from argschema import ArgSchema
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
    check_read_access,
    check_write_access,
    RaisingSchema,
)


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
    lfp = Nested(Lfp, many=False, required=True, allow_none=True)
    csd_path = String(required=True,
                      validate=check_read_access,
                      allow_none=True,
                      help="path to h5 file containing calculated current source density")
    sampling_rate = Float(default=30000.0, help="sampling rate (Hz, master clock) at which raw data were acquired on this probe")
    lfp_sampling_rate = Float(default=2500.0, allow_none=True, help="sampling rate of LFP data on this probe")
    temporal_subsampling_factor = Float(default=2.0, allow_none=True, help="subsampling factor applied to lfp data for this probe (across time)")
    spike_amplitudes_path = String(
        validate=check_read_access,
        help="path to npy file containing scale factor applied to the kilosort template used to extract each spike"
    )
    spike_templates_path = String(
        validate=check_read_access,
        help="path to file associating each spike with a kilosort template"
    )
    templates_path = String(
        validate=check_read_access,
        help="path to file contianing an (nTemplates)x(nSamples)x(nUnits) array of kilosort templates"
    )
    inverse_whitening_matrix_path = String(
        validate=check_read_access,
        help="Kilosort templates are whitened. In order to use them for scaling spike amplitudes to volts, we need to remove the whitening"
    )
    amplitude_scale_factor = Float(
        default=0.195e-6,
        help="amplitude scale factor converting raw amplitudes to Volts. Default converts from bits -> uV -> V"
    )


class InvalidEpoch(RaisingSchema):
    id = Int(required=True)
    type = String(required=True)
    label = String(required=True)
    start_time = Float(required=True)
    end_time = Float(required=True)


class SessionMetadata(RaisingSchema):
    specimen_name = String(required=True)
    age_in_days = Float(required=True)
    full_genotype = String(required=True)
    strain = String(required=True)
    sex = String(required=True)
    stimulus_name = String(required=True)
    species = String(required=True)
    donor_id = Int(required=True)


class InputSchema(ArgSchema):
    class Meta:
        unknown = mm.RAISE

    log_level = LogLevel(
        default="INFO", help="set the logging level of the module"
    )
    output_path = String(
        required=True,
        validate=check_write_access,
        help="write outputs to here",
    )
    session_id = Int(
        required=True, help="unique identifier for this ecephys session"
    )
    session_start_time = DateTime(
        required=True,
        help="the date and time (iso8601) at which the session started",
    )
    stimulus_table_path = String(
        required=True,
        validate=check_read_access,
        help="path to stimulus table file",
    )
    invalid_epochs = Nested(
        InvalidEpoch,
        many=True,
        required=True,
        help="epochs with invalid data"
    )
    probes = Nested(
        Probe,
        many=True,
        required=True,
        help="records of the individual probes used for this experiment",
    )
    running_speed_path = String(
        required=True,
        help="data collected about the running behavior of the experiment's subject",
    )
    session_sync_path = String(
        required=True,
        validate=check_read_access,
        help="Path to an h5 experiment session sync file (*.sync). This file relates events from different acquisition modalities to one another in time."
    )
    eye_tracking_rig_geometry = Dict(
        required=True,
        help="Mapping containing information about session rig geometry used for eye gaze mapping."
    )
    eye_dlc_ellipses_path = String(
        required=True,
        validate=check_read_access,
        help="h5 filepath containing raw ellipse fits produced by Deep Lab Cuts of subject eye, pupil, and corneal reflections during experiment"
    )
    eye_gaze_mapping_path = String(
        required=False,
        allow_none=True,
        help="h5 filepath containing eye gaze behavior of the experiment's subject"
    )
    pool_size = Int(
        default=3,
        help="number of child processes used to write probewise lfp files"
    )
    optotagging_table_path = String(
        required=False,
        validate=check_read_access,
        help="file at this path contains information about the optogenetic stimulation applied during this "
    )
    session_metadata = Nested(SessionMetadata, allow_none=True, required=False, help="miscellaneous information describing this session")


class ProbeOutputs(RaisingSchema):
    nwb_path = String(required=True)
    id = Int(required=True)


class OutputSchema(RaisingSchema):
    nwb_path = String(required=True, description='path to output file')
    probe_outputs = Nested(ProbeOutputs, required=True, many=True)
