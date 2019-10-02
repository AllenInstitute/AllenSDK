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

def test_BinaryIntervalSearchTree_shared_endpoint():
    
    bist = si.BinaryIntervalSearchTree([(0, 1, 'A'), (1, 2, 'B')])
    assert bist.search(0)[2] == 'A'
    assert bist.search(1)[2] == 'A'
    assert bist.search(1.5)[2] == 'B'

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


def test_pixels_to_visual_degrees():

    m = si.BrainObservatoryMonitor()

    np.testing.assert_almost_equal(m.pixels_to_visual_degrees(45), 4.64716996476)
    np.testing.assert_almost_equal(m.pixels_to_visual_degrees(45, small_angle_approximation=False), 4.64462483116)

    np.testing.assert_almost_equal(m.pixels_to_visual_degrees(1), 0.103270443661)
    np.testing.assert_almost_equal(m.pixels_to_visual_degrees(1, small_angle_approximation=False), 0.103270415704)

@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_lsn_image_to_screen(data_set):

    compare_set = set(data_set.list_stimuli()).intersection(si.LOCALLY_SPARSE_NOISE_STIMULUS_TYPES)
    if len(compare_set) > 0:
        for stimulus_type in compare_set:

            template = data_set.get_stimulus_template(stimulus_type)
            m = si.BrainObservatoryMonitor()
            m.lsn_image_to_screen(template[0,:,:]).shape == si.MONITOR_DIMENSIONS

@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_natural_movie_image_to_screen(data_set):

    compare_set = set(data_set.list_stimuli()).intersection(si.NATURAL_MOVIE_STIMULUS_TYPES)
    if len(compare_set) > 0:
        for stimulus_type in compare_set:

            template = data_set.get_stimulus_template(stimulus_type)
            m = si.BrainObservatoryMonitor()
            m.natural_movie_image_to_screen(template[0, :, :]).shape == si.MONITOR_DIMENSIONS

@pytest.mark.skipif(not os.path.exists('/projects/neuralcoding'),
                    reason="test NWB file not available")
def test_grating_to_screen(data_set):

    compare_set = set(data_set.list_stimuli()).intersection([si.STATIC_GRATINGS, si.DRIFTING_GRATINGS])
    if len(compare_set) > 0:

        for stimulus_type in compare_set:
            m = si.BrainObservatoryMonitor()
            curr_row = data_set.get_stimulus_table(stimulus_type).iloc[10]
            phase = 0
            spatial_frequency = .04
            orientation = curr_row.orientation
            template =  m.grating_to_screen(phase, spatial_frequency, orientation)
            assert m.natural_movie_image_to_screen(template).shape == si.MONITOR_DIMENSIONS

def test_get_mask():
    m = si.BrainObservatoryMonitor()
    mask = m.get_mask()

    assert mask.sum() == 931286
    assert mask.shape == si.MONITOR_DIMENSIONS


def test_mask():
    m = si.BrainObservatoryMonitor()

    assert(m._mask is None)

    assert(m.mask.sum() == 931286)
    assert(m.mask.shape == si.MONITOR_DIMENSIONS)
    assert(m._mask is not None)


def test_translate_image_and_fill():
    '''
    [[1 2 3]
    [4 5 6]
    [7 8 9]]

    [[127   4   5]
    [127   7   8]
    [127 127 127]]
    '''


    X = np.array([[1,2,3],[4,5,6],[7,8,9]])
    X_test = np.array([[127, 4, 5], [127, 7, 8], [127, 127, 127]])
    X_result = si.translate_image_and_fill(X, translation=(1,1))

    np.testing.assert_array_almost_equal(X_result, X_test)

def test_visual_degrees_to_pixels():
    
    m = si.BrainObservatoryMonitor()
    np.testing.assert_approx_equal(m.visual_degrees_to_pixels(4.5), 43.5749072092)

def test_spatial_frequency_to_pix_per_cycle():

    m = si.BrainObservatoryMonitor()

    x1 = m.spatial_frequency_to_pix_per_cycle(.1, 15.0)
    x2 = m.spatial_frequency_to_pix_per_cycle(.05, 15.0)

    np.testing.assert_almost_equal(x1, 97.7072500845)
    np.testing.assert_almost_equal(x2/x1, 2)

def test_show_image():

    m = si.BrainObservatoryMonitor()

    img = np.zeros(si.MONITOR_DIMENSIONS)
    m.show_image(img, show=False, warp=True, mask=False)
    m.show_image(img, show=False, warp=False, mask=True)

def test_map_stimulus():

    m = si.BrainObservatoryMonitor()
    test_list = [(0, 0), (-5.333333333333333, -7.333333333333333), (-5.333333333333333, -7.333333333333333), (-2.6666666666666665, -3.6666666666666665), (-16.88888888888889, 0.0), (-16.88888888888889, 0.0), (-16.88888888888889, 0.0), (-141.0, -373.0), (0, 0), (0, 0), (240.0, 330.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (50.666666666666664, 104.5), (50.666666666666664, 104.5), (50.666666666666664, 104.5), (99.0, -43.0), (240.0, 330.0), (240.0, 330.0), (240.0, 330.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (50.666666666666664, 104.5), (50.666666666666664, 104.5), (50.666666666666664, 104.5), (99.0, -43.0), (240.0, 330.0), (240.0, 330.0), (240.0, 330.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (50.666666666666664, 104.5), (50.666666666666664, 104.5), (50.666666666666664, 104.5), (99.0, -43.0), (240.0, 330.0), (240.0, 330.0), (60.0, 0.0), (-4.0, -7.333333333333333), (-4.0, -7.333333333333333), (-2.0, -3.6666666666666665), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-81.0, -373.0), (60.0, 0.0), (60.0, 0.0), (60.0, 0.0), (-4.0, -7.333333333333333), (-4.0, -7.333333333333333), (-2.0, -3.6666666666666665), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-81.0, -373.0), (60.0, 0.0), (60.0, 0.0), (60.0, 0.0), (-4.0, -7.333333333333333), (-4.0, -7.333333333333333), (-2.0, -3.6666666666666665), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-81.0, -373.0), (60.0, 0.0), (60.0, 0.0), (141.0, 373.0), (-2.2, 0.9555555555555556), (-2.2, 0.9555555555555556), (-1.1, 0.4777777777777778), (22.8, 118.11666666666666), (22.8, 118.11666666666666), (22.8, 118.11666666666666), (0.0, 0.0), (141.0, 373.0), (141.0, 373.0), (0, 0), (-5.333333333333333, -7.333333333333333), (-5.333333333333333, -7.333333333333333), (-2.6666666666666665, -3.6666666666666665), (-16.88888888888889, 0.0), (-16.88888888888889, 0.0), (-16.88888888888889, 0.0), (-141.0, -373.0), (0, 0), (0, 0), (0, 0), (-5.333333333333333, -7.333333333333333), (-5.333333333333333, -7.333333333333333), (-2.6666666666666665, -3.6666666666666665), (-16.88888888888889, 0.0), (-16.88888888888889, 0.0), (-16.88888888888889, 0.0), (-141.0, -373.0), (0, 0), (0, 0)]
    counter = 0
    for source_stimulus in sorted(si.all_stimuli()):
        for target_stimulus in sorted(si.all_stimuli()):
            tmp = m.map_stimulus((0,0), source_stimulus, target_stimulus)
            np.testing.assert_array_almost_equal(tmp, test_list[counter])
            counter += 1
            np.testing.assert_array_almost_equal(m.map_stimulus(tmp, target_stimulus, source_stimulus), np.array([0,0]))

if __name__ == "__main__":
#
    # with open(nwb_list_file, 'r') as f:
    #     NWB_FLAVORS = [l.strip() for l in f]
    #
    # for nwb_file_location in NWB_FLAVORS:
    #     data_set = BrainObservatoryNwbDataSet(nwb_file_location)
    #     test_lsn_image_to_screen(data_set)
    #     test_natural_movie_image_to_screen(data_set)
    #     test_grating_to_screen(data_set)

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
    # test_brain_observatory_monitor()
    # test_spatial_frequency_to_pix_per_cycle()
    # test_get_mask()
    # test_show_image()
    test_map_stimulus()