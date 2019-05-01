from marshmallow import RAISE, ValidationError

from argschema import ArgSchema, ArgSchemaParser
from argschema.schemas import DefaultSchema
from argschema.fields import LogLevel, String, Int, DateTime, Nested, Boolean, Float

from allensdk.brain_observatory.argschema_utilities import check_read_access, check_write_access, RaisingSchema
from allensdk.brain_observatory.nwb.schemas import RunningSpeedPathsSchema


class Channel(RaisingSchema):
    id = Int(required=True)
    probe_id = Int(required=True)
    valid_data = Boolean(required=True)
    local_index = Int(required=True)
    probe_vertical_position = Int(required=True)
    probe_horizontal_position = Int(required=True)
    manual_structure_id = Int(required=True, allow_none=True)
    manual_structure_acronym = String(required=True, allow_none=True)


class Unit(RaisingSchema):
    id = Int(required=True)
    peak_channel_id = Int(required=True)
    local_index = Int(required=True, description='within-probe index of this unit. Used for indexing into the spike times file.')
    quality = String(required=True)
    firing_rate = Float(required=True)
    snr = Float(required=True)
    isi_violations = Float(required=True)


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
    lfp = Nested(Lfp, many=False, required=True)


class InputSchema(ArgSchema):
    class Meta:
        unknown=RAISE
    log_level = LogLevel(default='INFO', description='set the logging level of the module')
    output_path = String(required=True, validate=check_write_access, description='write outputs to here')
    session_id = Int(required=True, description='unique identifier for this ecephys session')
    session_start_time = DateTime(required=True, description='the date and time (iso8601) at which the session started')
    stimulus_table_path = String(required=True, validate=check_read_access, description='path to stimulus table file')
    probes = Nested(Probe, many=True, required=True, description='records of the individual probes used for this experiment')
    running_speed = Nested(RunningSpeedPathsSchema, required=True, description='data collected about the running behavior of the experiment\'s subject')


class ProbeOutputs(RaisingSchema):
    nwb_path = String(required=True)
    id = Int(required=True)


class OutputSchema(RaisingSchema):
    nwb_path = String(required=True, description='path to output file')
    probe_nwb_paths = Nested(ProbeOutputs, required=True)