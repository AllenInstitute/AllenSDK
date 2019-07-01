from argschema import ArgSchema 
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, String, Float, Int, List, Bool
import numpy as np


class ProbeInputParameters(DefaultSchema):
    name = String(required=True, help='Identifier for this probe.')
    lfp_data_path = String(required=True, help='Path to lfp data for this probe')
    lfp_timestamps_path = String(required=True, help="Path to aligned lfp timestamps for this probe.")
    surface_channel = Int(required=True, help='Estimate of surface (pia boundary) channel index')
    reference_channels = List(Int, many=True, help='Indices of reference channels for this probe')
    csd_output_path = String(required=True, help='CSD output will be written here.')
    relative_window_output_path = String(required=True, help='Timestamp window surrounding each stimulus frame onset will be written here.')
    sampling_rate = Float(required=True, help='sampling rate assessed on master clock')
    total_channels = Int(default=384, help='Total channel count for this probe.')
    surface_channel_adjustment = Int(default=40, help='Erring up in the surface channel estimate is less dangerous for the CSD calculation than erring down, so an adjustment is provided.')
    spacing = Float(default=0.04, help='distance (in millimiters) between lengthwise-adjacent rows of recording sites on this probe.')


class StimulusInputParameters(DefaultSchema):
    stimulus_table_path = String(required=True, help='Path to stimulus table')
    key = String(required=True, help='CSD is calculated from a specific stimulus, defined (in part) by this key.')
    index = Int(default=None, allow_none=True, help='CSD is calculated from a specific stimulus, defined (in part) by this index.')


class InputParameters(ArgSchema):
    stimulus = Nested(StimulusInputParameters, required=True, help='Defines the stimulus from which CSD is calculated')
    probes = Nested(ProbeInputParameters, many=True, required=True, help='Probewise parameters.')
    pre_stimulus_time = Float(required=True, help='how much time pre stimulus onset is used for CSD calculation ')
    post_stimulus_time = Float(required=True, help='how much time post stimulus onset is used for CSD calculation ')
    num_trials = Int(default=None, allow_none=True, help='Number of trials after stimulus onset from which to compute CSD')
    volts_per_bit = Float(default=1.0, help='If the data are not in units of volts, they must be converted. In the past, this value was 0.195')
    memmap = Bool(default=True, help='whether to memory map the data file on disk or load it directly to main memory')
    memmap_thresh = Float(default=np.inf, help='files larger than this threshold (bytes) will be memmapped, regardless of the memmap setting.')


class ProbeOutputParameters(DefaultSchema):
    name = String(required=True, help='Identifier for this probe.')
    csd_path = String(required=True, help='Path to current source density file.')
    relative_window_path = String(required=True, help='Path to npy file containing time window around each stimulus onset.')
    csd_channels = List(Int, required=True, help='LFP channels from which CSD was calculated.')


class OutputSchema(DefaultSchema): 
    input_parameters = Nested(InputParameters, 
                              description=("Input parameters the module " 
                                           "was run with"), 
                              required=True) 


class OutputParameters(OutputSchema): 
    probe_outputs = Nested(ProbeOutputParameters, many=True, required=True, help='probewise outputs')
