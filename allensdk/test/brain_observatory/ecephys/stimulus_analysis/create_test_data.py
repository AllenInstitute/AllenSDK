import os
from argparse import ArgumentParser
import numpy as np

from allensdk.brain_observatory.ecephys.stimulus_analysis import \
    StaticGratings, \
    DriftingGratings, \
    NaturalScenes, \
    NaturalMovies, \
    Flashes, \
    DotMotion, \
    ReceptiveFieldMapping

OUTPUT_DIR = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh/expected'
OVERWRITE = True

#SPIKE_FILE = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh/data/mouse406807_integration_test.spikes.nwb2'
#col_opts = {'col_ori': 'ori', 'col_sf': 'sf', 'col_phase': 'phase', 'col_contrast': 'contrast', 'col_color': 'color'}

#SPIKE_FILE = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis/data/756029989_integration_test.spikes.nwb2'
#SPIKE_FILE = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh/data/756029989_integration_test.spikes.nwb2'

# SPIKE_FILE = '/allen/programs/braintv/workgroups/neuralcoding/Ephys_NWB_pilot/NWB_2_0/756029989_integration_test.spikes.nwb2'
# col_opts = {}

SPIKE_FILE = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh/data/ecephys_session_773418906.nwb'
col_opts = {}
# Since it would normally take 30+ mins to calculate metrics for all the units I've selected 30 units all from VISp
unit_id_filter = [914580630, 914580280, 914580278, 914580634, 914580610, 914580290, 914580288, 914580286, 914580284,
                  914580282, 914580294, 914580330, 914580304, 914580292, 914580300, 914580298, 914580308, 914580306,
                  914580302, 914580316, 914580314, 914580312, 914580310, 914580318, 914580324, 914580322, 914580320,
                  914580328, 914580326, 914580334]
avail_stims = [
    #'static_gratings',
    #'drifting_gratings',
    #'natural_scenes',
    #'natural_movies',
    #'flashes',
    #'dot_motion',
    'receptive_field_mapping'
]


stim_classes = {
    'static_gratings': StaticGratings,
    'drifting_gratings': DriftingGratings,
    'natural_scenes': NaturalScenes,
    'natural_movies': NaturalMovies,
    'flashes': Flashes,
    'dot_motion': DotMotion,
    'receptive_field_mapping': ReceptiveFieldMapping
}

np.random.seed(0)

def save_metrics_data(spikes_file, output_dir, stim_type, overwrite=True):
    mouse_id = os.path.basename(spikes_file).split('.')[0]

    metrics_file = os.path.join(output_dir, '{}.{}.csv'.format(mouse_id, stim_type))
    if not os.path.exists(metrics_file) or overwrite:
        analysis = stim_classes[stim_type](spikes_file, filter=unit_id_filter, **col_opts)
        analysis.metrics.to_csv(metrics_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--spikes-file', default=SPIKE_FILE)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--overwrite', action='store_true', default=OVERWRITE)
    args = parser.parse_args()

    for stim_type in avail_stims:
        save_metrics_data(spikes_file=args.spikes_file, output_dir=args.output_dir, stim_type=stim_type)



