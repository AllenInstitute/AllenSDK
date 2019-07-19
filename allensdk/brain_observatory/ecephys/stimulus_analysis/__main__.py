
from argschema import ArgSchemaParser
import time

import pandas as pd
import numpy as np

from functools import reduce

from ..ecephys_session import EcephysSession

from .drifting_gratings import DriftingGratings
from .static_gratings import StaticGratings
from .natural_scenes import NaturalScenes
from .natural_movies import NaturalMovies
from .dot_motion import DotMotion
from .flashes import Flashes
from .receptive_field_mapping import ReceptiveFieldMapping


def calculate_stimulus_metrics(args):

    print('ecephys: stimulus metrics module')

    start = time.time()

    stimulus_classes = (
                 #DriftingGratings,
                 #StaticGratings,
                 #NaturalScenes,
                 #NaturalMovies,
                 #DotMotion,
                 Flashes,
                 ReceptiveFieldMapping,
                )

    df = reduce(lambda output, nwb_path: \
                 pd.concat((output,
                           add_metrics_to_units_table(nwb_path, stimulus_classes, args))),
                 args['nwb_paths'],
                 pd.DataFrame()
                 )

    df.to_csv(args['output_file'])

    execution_time = time.time() - start

    print('total time: ' + str(np.around(execution_time, 2)) + ' seconds\n')

    return {"execution_time" : execution_time}


def add_metrics_to_units_table(nwb_path, stimulus_classes, args):

    """
    Adds columns to units table for one session, based on the metrics
    for each stimulus type.

    Parameters:
    -----------
    nwb_path : String
        Path to a spikes NWB file
    stimulus_classes : tuple
        Classes that add new columns to a units table

    Returns:
    --------
    units_df : pandas.DataFrame
        Units table with new columns appended

    """

    print(nwb_path)

    session = EcephysSession.from_nwb_path(nwb_path)

    metrics = [stim(session, params=args).metrics for stim in stimulus_classes]
    metrics.insert(0, session.units)

    return reduce(lambda left,right: pd.merge(left, right, on='unit_id'), metrics)


def main():

    from ._schemas import InputParameters, OutputParameters

    mod = ArgSchemaParser(schema_type=InputParameters,
                          output_schema_type=OutputParameters)

    output = calculate_stimulus_metrics(mod.args)

    output.update({"input_parameters": mod.args})

    if "output_json" in mod.args:
        mod.output(output, indent=2)
    else:
        print(mod.get_output_json(output))


if __name__ == "__main__":

    main()

