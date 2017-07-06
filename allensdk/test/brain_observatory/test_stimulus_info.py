import pytest
import numpy as np
import os
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet, si
import numpy as np
from pkg_resources import resource_filename  # @UnresolvedImport
NWB_FLAVORS = []

if 'TEST_NWB_FILES' in os.environ:
    nwb_list_file = os.environ['TEST_NWB_FILES']
else:
    nwb_list_file = resource_filename(__name__, os.path.join('..','core','nwb_files.txt'))

if nwb_list_file == 'skip':
    NWB_FLAVORS = []
else:
    with open(nwb_list_file, 'r') as f:
        NWB_FLAVORS = [l.strip() for l in f]

@pytest.fixture(params=NWB_FLAVORS)
def data_set(request):
    data_set = BrainObservatoryNwbDataSet(request.param)

    return data_set

def test_BinaryIntervalSearchTree():

    bist = si.BinaryIntervalSearchTree([(0, .9, 'A'), (1, 1.9, 'B'), (3, 3.9, 'D'), (2, 2.9, 'C')])
    assert bist.search(1.5)[2] == 'B'
    assert bist.search(0)[2] == 'A'
    assert bist.search(2.5)[2] == 'C'
    assert bist.search(3.5)[2] == 'D'

def test_pixels_to_visual_degrees():
    m = si.BrainObservatoryMonitor()
    np.testing.assert_almost_equal(m.pixels_to_visual_degrees(1), 0.103270443661,10)

@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_StimulusSearch(data_set):

    epoch_df = data_set.get_stimulus_epoch_table()
    s = si.StimulusSearch(data_set)
    assert len(s.search(epoch_df.iloc[2]['end'])) == 3
    assert s.search(epoch_df.iloc[2]['end'] + 1) is None
    assert len(s.search(752)) == 3


def test_sessions_with_stimulus():

    for session_type, stimulus_type_list in si.SESSION_STIMULUS_MAP.items():
        for stimulus_type in stimulus_type_list:
            assert session_type in si.sessions_with_stimulus(stimulus_type)

def test_stimuli_in_session():

    for session_type, stimulus_type_list in si.SESSION_STIMULUS_MAP.items():
        for stimulus_type in stimulus_type_list:
            assert session_type in si.sessions_with_stimulus(stimulus_type)

def test_stimuli_in_session():

    test_dict = {si.THREE_SESSION_A:4,
                 si.THREE_SESSION_B:4,
                 si.THREE_SESSION_C:4,
                 si.THREE_SESSION_C2:5}

    for session in si.SESSION_LIST:
        assert session in si.SESSION_STIMULUS_MAP
        assert len(si.stimuli_in_session(session)) == test_dict[session]
    assert len(si.SESSION_STIMULUS_MAP) == len(si.SESSION_LIST) == 4

def test_all_stimuli():
    assert len(si.all_stimuli()) == 10

def test_rotate():
    np.testing.assert_array_almost_equal(np.array(si.rotate(1,1,np.pi)), np.array([-1,-1]))

def test_get_spatial_grating():

    data = si.get_spatial_grating(height=100, aspect_ratio=2, ori=45, pix_per_cycle=10, phase=0, p2p_amp=2, baseline=1)

    assert data.shape == (100,200)
    np.testing.assert_almost_equal(data[0,0], data[-1,-1])
    np.testing.assert_almost_equal(data.max(), 2, 3)
    np.testing.assert_almost_equal(data.min(), 0, 3)
    np.testing.assert_almost_equal(data[50,100], 2)

def test_get_spatio_temporal_grating():

    for t, test_val in zip([0,.5,1], [2,0,2]):
        data = si.get_spatio_temporal_grating(t, height=100, aspect_ratio=2, ori=45, pix_per_cycle=10, phase=0, p2p_amp=2, baseline=1, temporal_frequency=1)
        np.testing.assert_almost_equal(data[50,100], test_val)

    data = si.get_spatio_temporal_grating(0, height=100,
                                             aspect_ratio=2,
                                             ori=45,
                                             pix_per_cycle=20,
                                             phase=0,
                                             p2p_amp=2,
                                             baseline=1,
                                             temporal_frequency=1)

    x1 = data[50, 100]

    data = si.get_spatio_temporal_grating(.5, height=100,
                                             aspect_ratio=2,
                                             ori=45,
                                             pix_per_cycle=20,
                                             phase=.5,
                                             p2p_amp=2,
                                             baseline=1,
                                             temporal_frequency=1)

    x2 = data[50, 100]

    np.testing.assert_almost_equal(x1, x2)

def test_map_template_monitor():



    np.testing.assert_almost_equal(np.array((500, 250)),
                                   si.map_template_coordinate_to_monitor_coordinate((20, 20), (1000, 500), (40, 40)))


    np.testing.assert_almost_equal(np.array((20,20)),
                                   si.map_monitor_coordinate_to_template_coordinate((500, 250), (1000, 500), (40,40)))

def test_lsn_monitor():

    lsn4_template_coordinate = (8, 14)
    lsn4_monitor_coordinate = np.array(si.MONITOR_DIMENSIONS)/2 #(600,960)
    np.testing.assert_almost_equal(np.array(lsn4_monitor_coordinate),
                                   si.map_stimulus_coordinate_to_monitor_coordinate(lsn4_template_coordinate, si.MONITOR_DIMENSIONS, si.LOCALLY_SPARSE_NOISE_4DEG))

    lsn4_template_coordinate = (4, 7)
    lsn4_monitor_coordinate = np.array(si.MONITOR_DIMENSIONS)/2#(600,960)
    np.testing.assert_almost_equal(np.array(lsn4_monitor_coordinate),
                                   si.map_stimulus_coordinate_to_monitor_coordinate(lsn4_template_coordinate, si.MONITOR_DIMENSIONS, si.LOCALLY_SPARSE_NOISE_8DEG))

    lsn4_template_coordinate = (0,0)
    lsn4_monitor_coordinate = (240,330)
    np.testing.assert_almost_equal(np.array(lsn4_monitor_coordinate),
                                   si.map_stimulus_coordinate_to_monitor_coordinate(lsn4_template_coordinate,
                                                                           si.MONITOR_DIMENSIONS,
                                                                           si.LOCALLY_SPARSE_NOISE_4DEG))

    lsn4_template_coordinate = (0,0)
    lsn4_monitor_coordinate = (240,330)
    np.testing.assert_almost_equal(np.array(lsn4_template_coordinate),
                                   si.monitor_coordinate_to_lsn_coordinate(lsn4_monitor_coordinate,
                                                                           si.MONITOR_DIMENSIONS,
                                                                           si.LOCALLY_SPARSE_NOISE_4DEG))


    lsn4_template_coordinate = (0,0)
    lsn4_monitor_coordinate = (240,330)
    np.testing.assert_almost_equal(np.array(lsn4_monitor_coordinate),
                                   si.map_stimulus_coordinate_to_monitor_coordinate(lsn4_template_coordinate,
                                                                           si.MONITOR_DIMENSIONS,
                                                                           si.LOCALLY_SPARSE_NOISE_8DEG))

    lsn4_template_coordinate = (0,0)
    lsn4_monitor_coordinate = (240,330)
    np.testing.assert_almost_equal(np.array(lsn4_template_coordinate),
                                   si.monitor_coordinate_to_lsn_coordinate(lsn4_monitor_coordinate,
                                                                           si.MONITOR_DIMENSIONS,
                                                                           si.LOCALLY_SPARSE_NOISE_8DEG))

def test_natural_scene_monitor():

    template_coordinate = (0,0)
    monitor_coordinate = (141, 373)
    np.testing.assert_almost_equal(np.array(monitor_coordinate),
                                   si.natural_scene_coordinate_to_monitor_coordinate(template_coordinate,
                                                                                     si.MONITOR_DIMENSIONS))

    template_coordinate = (0,0)
    monitor_coordinate = (141, 373)
    np.testing.assert_almost_equal(np.array(template_coordinate),
                                   si.map_monitor_coordinate_to_stimulus_coordinate(monitor_coordinate,
                                                                                    si.MONITOR_DIMENSIONS,
                                                                                    si.NATURAL_SCENES))

def test_natural_movie_monitor():

    template_coordinate = (0,0)
    monitor_coordinate = (60, 0)
    np.testing.assert_almost_equal(np.array(monitor_coordinate),
                                   si.natural_movie_coordinate_to_monitor_coordinate(template_coordinate,
                                                                                     si.MONITOR_DIMENSIONS))

    template_coordinate = (0,0)
    monitor_coordinate = (60, 0)
    np.testing.assert_almost_equal(np.array(template_coordinate),
                                   si.map_monitor_coordinate_to_stimulus_coordinate(monitor_coordinate,
                                                                                    si.MONITOR_DIMENSIONS,
                                                                                    si.NATURAL_MOVIE_ONE))

def test_bijective_all_stimuli():

    for stimulus in si.all_stimuli():

        template_coordinate = (10,10)
        monitor_coordinate = si.map_stimulus_coordinate_to_monitor_coordinate(template_coordinate,
                                                                              si.MONITOR_DIMENSIONS,
                                                                              stimulus)

        new_template_coordinate = si.map_monitor_coordinate_to_stimulus_coordinate(monitor_coordinate,
                                                                                   si.MONITOR_DIMENSIONS,
                                                                                   stimulus)

        np.testing.assert_array_almost_equal(template_coordinate, new_template_coordinate)

        for original_loc in [(0,0), (10,10)]:

            for target_stimulus in si.all_stimuli():

                new_loc = si.map_stimulus(original_loc, stimulus, target_stimulus, si.MONITOR_DIMENSIONS)
                new_original_loc = si.map_stimulus(new_loc, target_stimulus, stimulus, si.MONITOR_DIMENSIONS)
                np.testing.assert_array_almost_equal(new_original_loc, original_loc)

def test_monitor_basic_spatial_unit():

    m = si.Monitor(300,400, 5, 'cm')
    m.set_spatial_unit('cm')

    m.set_spatial_unit('inch')
    np.testing.assert_almost_equal(m.panel_size, 1.968505, 5)
    np.testing.assert_almost_equal(1./m.aspect_ratio, 3./4)
    np.testing.assert_almost_equal(m.height, 0.46500143220300011)
    np.testing.assert_almost_equal(m.width, 0.62000190960400015)
    np.testing.assert_almost_equal(m.pixel_size, 0.0015500047740100004)

    m.set_spatial_unit('cm')
    np.testing.assert_almost_equal(m.panel_size, 5)
    np.testing.assert_almost_equal(1./m.aspect_ratio, 3./4)
    np.testing.assert_almost_equal(m.height, 3)
    np.testing.assert_almost_equal(m.width, 4)
    np.testing.assert_almost_equal(m.pixel_size, .01)












if __name__ == "__main__":

    # with open(nwb_list_file, 'r') as f:
    #     NWB_FLAVORS = [l.strip() for l in f]
    #
    # print NWB_FLAVORS
    # data_set = BrainObservatoryNwbDataSet(request.param)
    # test_StimulusSearch()
    # test_BinaryIntervalSearchTree()

    # test_sessions_with_stimulus()
    # test_stimuli_in_session()
    # test_all_stimuli()
    # test_rotate()
    # test_get_spatial_grating()
    # test_get_spatio_temporal_grating()
    # test_map_template_coordinate_to_monitor_coordinate()
    # test_natural_scene_monitor()
    # test_bijective_all_stimuli()
    # test_monitor_basic_spatial_unit()
    test_brain_observatory_monitor()