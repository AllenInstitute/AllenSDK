from marshmallow import RAISE

from argschema import ArgSchema
from argschema.fields import (
    LogLevel,
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
    id = Int(required=True)
    probe_id = Int(required=True)
    valid_data = Boolean(required=True)
    local_index = Int(required=True)
    probe_vertical_position = Int(required=True)
    probe_horizontal_position = Int(required=True)
    structure_id = Int(required=True, allow_none=True)
    structure_acronym = String(required=True, allow_none=True)
    AP_coordinate = Float(required=True, allow_none =True)
    DV_coordinate = Float(required=True, allow_none= True)
    ML_coordinate = Float(required=True, allow_none=True)
    cortical_depth = Float(required=True, allow_none=True)

class Unit(RaisingSchema):
    id = Int(required=True)
    peak_channel_id = Int(required=True)
    local_index = Int(
        required=True,
        help="within-probe index of this unit. Used for indexing into the spike times file.",
    )
    quality = String(required=True)
    firing_rate = Float(required=True)
    snr = Float(required=True)
    isi_violations = Float(required=True)
    presence_ratio = Float(required=True)
    amplitude_cutoff = Float(required=True)
    isolation_distance = Float(required=True)
    l_ratio = Float(required=True)
    d_prime = Float(required=True)
    nn_hit_rate = Float(required=True)
    nn_miss_rate = Float(required=True)


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
<<<<<<< HEAD
    #lfp = Nested(Lfp, many=False, required=True)
=======
    lfp = Nested(Lfp, many=False, required=True)
    csd_path = String(required=True, validate=check_read_access, help="path to h5 file containing calculated current source density")
>>>>>>> 84d50266004514d15660f22dd91f2f616513d3fe


class InputSchema(ArgSchema):
    class Meta:
        unknown = RAISE

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
    pool_size = Int(
        default=3, 
        help="number of child processes used to write probewise lfp files"
    )


class ProbeOutputs(RaisingSchema):
    nwb_path = String(required=True)
    id = Int(required=True)


class OutputSchema(RaisingSchema):
    nwb_path = String(required=True, description='path to output file')
    #probe_outputs = Nested(ProbeOutputs, required=True, many=True)