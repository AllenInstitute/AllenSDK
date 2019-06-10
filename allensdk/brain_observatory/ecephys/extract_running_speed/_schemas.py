from argschema import ArgSchema, ArgSchemaParser 
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, InputDir, String, Float, Dict, Int


class InputParameters(ArgSchema): 
    stimulus_pkl_path = String(required=True, help="path to pkl file containing raw stimulus information")
    sync_h5_path = String(required=True, help="path to h5 file containing synchronization information")
    wheel_radius = Float(default=5.5036, help='radius, in cm, of running wheel')
    output_running_speeds_path = String(required=True, help="the output running speeds file will be written here")
    output_timestamps_path = String(required=True, help='the start times of the frames at which running speeds were sampled (global clock) will be written here')
    frame_time_strategy = String(default='use_photodiode', 
        help='technique used to align frame times. Options are \'use_photodiode\', which interpolates frame times between photodiode edge times (preferred when vsync times are unreliable) and \'use_vsyncs\', which is preferred when reliable vsync times are available.'
    )


class OutputSchema(DefaultSchema): 
    input_parameters = Nested(InputParameters, 
                              description=("Input parameters the module " 
                                           "was run with"), 
                              required=True) 


class OutputParameters(OutputSchema): 
    output_running_speeds_path = String(required=True, help="the output running speeds file was  written here")
    output_timestamps_path = String(required=True, help='the times at which running speeds were sampled (global clock) were written here')