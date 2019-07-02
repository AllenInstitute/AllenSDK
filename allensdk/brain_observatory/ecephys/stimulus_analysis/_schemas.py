from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, String, Float, List, Int


class DriftingGratings(DefaultSchema):

    stimulus_key = String(help='Key for the drifting gratings stimulus')


class StaticGratings(DefaultSchema):

    stimulus_key = String(help='Key for the static gratings stimulus')


class NaturalScenes(DefaultSchema):

    stimulus_key = String(help='Key for the natural scenes stimulus')


class NaturalMovies(DefaultSchema):

    stimulus_key = String(help='Key for the natural movies stimulus')


class DotMotion(DefaultSchema):

    stimulus_key = String(help='Key for the dot motion stimulus')


class ContrastTuning(DefaultSchema):

    stimulus_key = String(help='Key for the contrast tuning stimulus')


class Flashes(DefaultSchema):

    stimulus_key = String(help='Key for the flash stimulus')


class ReceptiveFieldMapping(DefaultSchema):

    stimulus_key = String(help='Key for the receptive field mapping stimulus')
    minimum_spike_count = Int(help='Minimum number of spikes for computing receptive field parameters')
    spatial_p_value_n_iter = Int(help='number of iterations for computing spatial p value')
    mask_threshold = Float(help='Threshold (as fraction of peak) for computing receptive field mask')



class InputParameters(ArgSchema):

    drifting_gratings = Nested(DriftingGratings)
    static_gratings = Nested(StaticGratings)
    natural_scenes = Nested(NaturalScenes)
    natural_movies = Nested(NaturalMovies)
    dot_motion = Nested(DotMotion)
    contrast_tuning = Nested(ContrastTuning)
    flashes = Nested(Flashes)
    receptive_field_mapping = Nested(ReceptiveFieldMapping)

    nwb_paths = List(String, help='List of paths to EcephysSession NWB files')
    output_file = String(help = 'Location for saving output file')


class OutputSchema(DefaultSchema):
    input_parameters = Nested(InputParameters,
                              description=("Input parameters the module "
                                           "was run with"),
                              required=True)

class OutputParameters(OutputSchema):

    execution_time = Float()
