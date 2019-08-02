import os
from argparse import ArgumentParser

from allensdk.brain_observatory.ecephys.stimulus_analysis import \
    StaticGratings, \
    DriftingGratings, \
    NaturalScenes, \
    NaturalMovies, \
    Flashes, \
    DotMotion, \
    ReceptiveFieldMapping

OUTPUT_DIR = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh/expected'

SPIKE_FILE = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh/data/mouse406807_integration_test.spikes.nwb2'
col_opts = {'col_ori': 'ori', 'col_sf': 'sf', 'col_phase': 'phase'}

#SPIKE_FILE = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis/data/756029989_integration_test.spikes.nwb2'
#SPIKE_FILE = '/allen/aibs/informatics/module_test_data/ecephys/stimulus_analysis_fh/data/756029989_integration_test.spikes.nwb2'
# SPIKE_FILE = '/allen/programs/braintv/workgroups/neuralcoding/Ephys_NWB_pilot/NWB_2_0/756029989_integration_test.spikes.nwb2'
# col_opts = {}


def save_metrics_data(spikes_file, output_dir, overwrite=False):
    mouse_id = os.path.basename(spikes_file).split('.')[0]

    metrics_file = os.path.join(output_dir, '{}.static_gratings.csv'.format(mouse_id))
    if not os.path.exists(metrics_file) or overwrite:
        sg = StaticGratings(spikes_file, **col_opts)
        sg.metrics.to_csv(metrics_file)

    metrics_file = os.path.join(output_dir, '{}.drifting_gratings.csv'.format(mouse_id))
    if not os.path.exists(metrics_file) or overwrite:
        dg = DriftingGratings(spikes_file, **col_opts)
        dg.metrics.to_csv(metrics_file)

    metrics_file = os.path.join(output_dir, '{}.natural_scenes.csv'.format(mouse_id))
    if not os.path.exists(metrics_file) or overwrite:
        ns = NaturalScenes(spikes_file)
        ns.metrics.to_csv(metrics_file)

    """
    metrics_file = os.path.join(output_dir, '{}.natural_movies.csv'.format(mouse_id))
    if not os.path.exists(metrics_file) or overwrite:
        nm = NaturalMovies(spikes_file, stimulus_names='natural_movie_one')
        nm.metrics.to_csv(metrics_file)
    """

    metrics_file = os.path.join(output_dir, '{}.flashes.csv'.format(mouse_id))
    if not os.path.exists(metrics_file) or overwrite:
        flashes = Flashes(spikes_file)
        flashes.metrics.to_csv(metrics_file)

    """
    metrics_file = os.path.join(output_dir, '{}.dot_motion.csv'.format(mouse_id))
    if not os.path.exists(metrics_file) or overwrite:
        dm = DotMotion(spikes_file)
        dm.metrics.to_csv(metrics_file)
    """

    metrics_file = os.path.join(output_dir, '{}.receptive_field_mapping.csv'.format(mouse_id))
    if not os.path.exists(metrics_file) or overwrite:
        rfm = ReceptiveFieldMapping(spikes_file)
        rfm.metrics.to_csv(metrics_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--spikes-file', default=SPIKE_FILE)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()

    save_metrics_data(args.spikes_file, args.output_dir)



