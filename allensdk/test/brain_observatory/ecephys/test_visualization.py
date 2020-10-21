import allensdk.brain_observatory.ecephys.visualization.__init__ as vis
import pandas as pd

def test_raster_plot():
    spike_times = pd.DataFrame({
        'unit_id': [2, 1, 2],
        'stimulus_presentation_id': [2, 2, 2, ],
        'time_since_stimulus_presentation_onset': [0.01, 0.02, 0.03]
    }, index=pd.Index(name='spike_time', data=[1.01, 1.02, 1.03]))
    
    fig = vis.raster_plot(spike_times)
    ax = fig.get_axes()[0]

    assert len(spike_times['unit_id'].unique()) == len(ax.collections)
