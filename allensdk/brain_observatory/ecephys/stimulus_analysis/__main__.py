from argschema import ArgSchemaParser
import time
import os
import pathlib
import pandas as pd
import numpy as np
import logging

from ..ecephys_session import EcephysSession
from .drifting_gratings import DriftingGratings
from .static_gratings import StaticGratings
from .natural_scenes import NaturalScenes
from .natural_movies import NaturalMovies
from .dot_motion import DotMotion
from .flashes import Flashes
from .receptive_field_mapping import ReceptiveFieldMapping

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
    barrier = comm.Barrier
except ModuleNotFoundError as e:
    MPI_rank = 0
    MPI_size = 1
    barrier = lambda : None


logger = logging.getLogger(__name__)


# Map between json file subsections and StimAnalysis subclass
# TODO: Try to order this list by how long each subclass takes to finish.
stim_classes = [
    ('receptive_field_mapping', ReceptiveFieldMapping),
    ('drifting_gratings', DriftingGratings),
    ('dot_motion', DotMotion),
    ('static_gratings', StaticGratings),
    ('natural_scenes', NaturalScenes),
    ('natural_moves', NaturalMovies),
    ('flashes', Flashes),
]


def log_info(message, all_ranks=False):
    if all_ranks or MPI_rank == 0:
        logger.info(message)


def load_session(nwb_path, stimulus_class, **session_params):
    session = EcephysSession.from_nwb_path(nwb_path, api_kwargs={
        "amplitude_cutoff_maximum": np.inf,
        "presence_ratio_minimum": -np.inf,
        "isi_violations_maximum": np.inf,
        "filter_by_validity": False  # actually you probably still want this one
    })
    return stimulus_class(session, **session_params)


def calculate_stimulus_metrics(args):
    log_info('ecephys: stimulus metrics module')
    start = time.time()

    input_session_nwb = args['input_session_nwb']
    output_file = args['output_file']

    # For each stimulus class that needs to be processed; calculate and save the metrics on a different rank (unless
    # MPI_size is small and one rank has to process two or more metrics).
    def _temp_csv_file(stim_class):
        # filename to save temporary stim_analysis csv files before being merged into final
        output_dir = pathlib.Path(output_file).parents[0]
        session_name = pathlib.Path(input_session_nwb).stem
        return os.path.join(output_dir, '{}.{}.csv'.format(session_name, stim_class))

    relevant_stim_class = [(sc[0], sc[1], _temp_csv_file(sc[0]))
                           for sc in stim_classes if sc[0] in args]  # only stims specified in the input json
    for sc_name, stim_class, tmp_csv in relevant_stim_class[MPI_rank::MPI_size]:
        analysis_obj = load_session(input_session_nwb, stim_class, **args[sc_name])
        # analysis_obj = stim_class(input_session_nwb, **args[sc_name])
        analysis_obj.metrics.to_csv(tmp_csv)

    barrier()  # wait till all the csv files have been created

    # Have the first rank go through all the created csv files and merge into one
    if MPI_rank == 0:
        final_table = pd.read_csv(relevant_stim_class[0][2])
        for _, _, tmp_csv in relevant_stim_class[1:]:
            tmp_table = pd.read_csv(tmp_csv)
            final_table = pd.merge(final_table, tmp_table, on='unit_id')

        final_table.to_csv(output_file)

        # Delete the temporary files
        for _, _, tmp_csv in relevant_stim_class:
            if os.path.exists(tmp_csv):
                try:
                    os.remove(tmp_csv)
                except Exception as e:
                    pass

    barrier()

    execution_time = time.time() - start
    log_info(f'total time: {str(np.around(execution_time, 2))} seconds')
    return {"execution_time" : execution_time}


def calculate_stimulus_metrics_inmem(args):
    # Same as above but use comm.Send instead of saving the individual csv files
    pass


def main():
    from ._schemas import InputParameters, OutputParameters

    mod = ArgSchemaParser(schema_type=InputParameters, output_schema_type=OutputParameters)
    output = calculate_stimulus_metrics(mod.args)
    if MPI_rank == 0:
        output.update({"input_parameters": mod.args})
        if "output_json" in mod.args:
            mod.output(output, indent=2)
        else:
            log_info(mod.get_output_json(output))
    barrier()


if __name__ == "__main__":
    main()
