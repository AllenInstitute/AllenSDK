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
    sampling_rate = Float(required=True, help='sampling rate assessed on master clock')
    total_channels = Int(default=384, help='Total channel count for this probe.')
    surface_channel_adjustment = Int(default=40, help='Erring up in the surface channel estimate is less dangerous for the CSD calculation than erring down, so an adjustment is provided.')
    spacing = Float(default=0.04, help='distance (in millimiters) between lengthwise-adjacent rows of recording sites on this probe.')
    phase = String(required=True, help='The probe type (3a or PXI) which determines if channels need to be reordered')


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
    memmap = Bool(default=False, help='whether to memory map the data file on disk or load it directly to main memory')
    memmap_thresh = Float(default=np.inf, help='files larger than this threshold (bytes) will be memmapped, regardless of the memmap setting.')
    filter_cuts = List(Float, default=[5.0, 150.0], cli_as_single_argument=True, help='Cutoff frequencies for bandpass filter')
    filter_order = Int(default=5, help='Order for bandpass filter')
    reorder_channels = Bool(default=True, help='Determines whether LFP channels should be re-ordered')
    noisy_channel_threshold = Float(default=1500.0, help='Threshold for removing noisy channels from analysis')


class ProbeOutputParameters(DefaultSchema):
    name = String(required=True, help='Identifier for this probe.')
    csd_path = String(required=True, help='Path to current source density file.')
    csd_channels = List(Int, required=True, help='LFP channels from which CSD was calculated.')


class OutputSchema(DefaultSchema):
    input_parameters = Nested(InputParameters,
                              description=("Input parameters the module "
                                           "was run with"),
                              required=True)


class OutputParameters(OutputSchema):
    stimulus_name = String(required=True, help="name of stimulus from which CSD was calculated")
    stimulus_index = Int(required=True, help="index of stimulus from which CSD was calculated")
    probe_outputs = Nested(ProbeOutputParameters, many=True, required=True, help='probewise outputs')
