"""
Traces with Stimulus Epochs
=================================

This is an example of how to show timeseries traces (dFF and running
speed) with stimulus epochs overlaid.
"""
# sphinx_gallery_thumbnail_number = 2
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.stimulus_info import (NATURAL_MOVIE_ONE_COLOR,
                                                      SPONTANEOUS_ACTIVITY_COLOR,
                                                      STATIC_GRATINGS_COLOR,
                                                      NATURAL_SCENES_COLOR)
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms

#####################
# Create a lookup table of stimulus name to a color for convenience as well
# as a helper function to color an axis based on stimulus presented.
stim_color = {
    "static_gratings": STATIC_GRATINGS_COLOR,
    "natural_scenes": NATURAL_SCENES_COLOR,
    "spontaneous": SPONTANEOUS_ACTIVITY_COLOR,
    "natural_movie_one": NATURAL_MOVIE_ONE_COLOR,
}

def color_axis_by_stimuli(ax, epoch_df):
    # a transform to make the fill go between data in x and between axes in y
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    stimuli_shaded = {} # used to track what has already been labeled
    for i, row in epoch_df.iterrows():
        stim_name = row['stimulus']
        if stimuli_shaded.get(stim_name, False):
            label = None
        else:
            label = stim_name
            stimuli_shaded[stim_name] = True
        color = stim_color[stim_name]
        start = time[row["start"]]
        end = time[row["end"]]
        ax.fill_between(time, 0, 1, where=((time >= start) & (time <= end)),
                        facecolor=color, alpha=0.5, transform=trans,
                        label=label)
    return ax

#####################
# Open a cache for downloading experiments using the API and grab an example
# experiment, and the id for the first cell specimen in the dataset, along
# with the stimulus table for the experiment.
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
experiment_id = 569407590
data_set = boc.get_ophys_experiment_data(experiment_id)
cell_id = data_set.get_cell_specimen_ids()[0]
epoch_df = data_set.get_stimulus_epoch_table()

#####################
# Plot the dF over F for the experiment with an overlay indicating the stimulus
# presented at the time. The get_dff_traces returns a (time, traces) tuple.

time, dff = data_set.get_dff_traces(cell_specimen_ids=[cell_id])

fig, ax = plt.subplots()
ax.plot(time, dff.T)
stimuli_shaded = {}
color_axis_by_stimuli(ax, epoch_df)
plt.title("dF/F")
plt.xlabel("time (s)")
plt.legend()
plt.show()

#####################
# Plot the running speed of the mouse with an overlay indicating the stimulus.
# The get_running_speed method returns a (speed, time) tuple.
speed, time = data_set.get_running_speed()

fig, ax = plt.subplots()
ax.plot(time, speed)
stimuli_shaded = {}
color_axis_by_stimuli(ax, epoch_df)
plt.title("running speed")
plt.ylabel("speed (cm/s)")
plt.xlabel("time (s)")
plt.legend()
plt.show()
