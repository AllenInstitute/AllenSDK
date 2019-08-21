import sys

from argschema import ArgSchema, ArgSchemaParser
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, InputDir, String, Float, Dict, Int, List, Bool

from . import naming_utilities as nu


class InputParameters(ArgSchema):
    stimulus_pkl_path = String(
        required=True, help="path to pkl file containing raw stimulus information"
    )
    sync_h5_path = String(
        required=True, help="path to h5 file containing syncronization information"
    )
    output_stimulus_table_path = String(
        required=True, help="the output stimulus table csv will be written here"
    )
    output_frame_times_path = String(required=True, help="output all frame times here")
    minimum_spontaneous_activity_duration = Float(
        default=sys.float_info.epsilon,
        help="detected spontaneous activity sweeps will be rejected if they last fewer that this many seconds",
    )
    maximum_expected_spontanous_activity_duration = Float(
        default=1225.02541,
        help="validation will fail if a spontanous activity epoch longer than this one is computed.",
    )
    frame_time_strategy = String(
        default="use_photodiode",
        help="technique used to align frame times. Options are 'use_photodiode', which interpolates frame times between photodiode edge times (preferred when vsync times are unreliable) and 'use_vsyncs', which is preferred when reliable vsync times are available.",
    )
    stimulus_name_map = Dict(
        keys=String(),
        values=String(),
        help="optionally rename stimuli",
        default={
            "": "spontaneous",

            "natural_movie_1" : "natural_movie_one",
            "natural_movie_3" : "natural_movie_three",
            "Natural Images": "natural_scenes",
            "flash_250ms": "flashes",
            "gabor_20_deg_250ms": "receptive_field_mapping",
            "drifting_gratings" : "drifting_gratings",
            "static_gratings" : "static_gratings",

            "contrast_response": "drifting_gratings_contrast",
            "natural_movie_1_more_repeats" : "natural_movie_one",
            "natural_movie_shuffled" : "natural_movie_one_shuffled",
            "motion_stimulus" : "dot_motion",
            "drifting_gratings_more_repeats" : "drifting_gratings_75_repeats",
            
            "signal_noise_test_0_200_repeats": "test_movie_one",

            "signal_noise_test_0": "test_movie_one",
            "signal_noise_test_0": "test_movie_two",
            "signal_noise_session_1" : "dense_movie_one",
            "signal_noise_session_2" : "dense_movie_two",
            "signal_noise_session_3" : "dense_movie_three",
            "signal_noise_session_4" : "dense_movie_four",
            "signal_noise_session_5" : "dense_movie_five",
            "signal_noise_session_6" : "dense_movie_six",

        },
    )
    column_name_map = Dict(
        keys=String(), values=String(), help="optionally rename parameters", default={}
    )
    extract_const_params_from_repr = Bool(default=True)
    drop_const_params = List(
        String(),
        help="columns to be dropped from the stimulus table",
        default=["name", "maskParams", "win", "autoLog", "autoDraw"],
    )


class OutputSchema(DefaultSchema):
    input_parameters = Nested(
        InputParameters,
        description=("Input parameters the module " "was run with"),
        required=True,
    )


class OutputParameters(OutputSchema):
    output_path = String(help="Path to output csv file")
    output_frame_times_path = String(help="output all frame times here")
