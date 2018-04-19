import pytest
import os
import tempfile
import logging
logging.basicConfig(level=logging.DEBUG)

from allensdk.brain_observatory.drifting_gratings import DriftingGratings
from allensdk.brain_observatory.static_gratings import StaticGratings
from allensdk.brain_observatory.natural_movie import NaturalMovie
from allensdk.brain_observatory.natural_scenes import NaturalScenes
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise
from allensdk.brain_observatory.session_analysis import SessionAnalysis
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet as BODS
import allensdk.brain_observatory.stimulus_info as si

@pytest.fixture()
def paths():
    return {
        'analysis_a': '/allen/aibs/informatics/module_test_data/observatory/plots/510859641_three_session_A_analysis.h5',
        'analysis_b': '/allen/aibs/informatics/module_test_data/observatory/plots/510698988_three_session_B_analysis.h5',
        'analysis_c': '/allen/aibs/informatics/module_test_data/observatory/plots/510532780_three_session_C_analysis.h5',
        'nwb_a': '/allen/aibs/informatics/module_test_data/observatory/plots/510859641.nwb',
        'nwb_b': '/allen/aibs/informatics/module_test_data/observatory/plots/510698988.nwb',
        'nwb_c': '/allen/aibs/informatics/module_test_data/observatory/plots/510532780.nwb'
    }

@pytest.fixture()
def nwb_a(paths):
    return paths['nwb_a']

@pytest.fixture()
def nwb_b(paths):
    return paths['nwb_b']

@pytest.fixture()
def nwb_c(paths):
    return paths['nwb_c']

@pytest.fixture()
def analysis_a(paths):
    return paths['analysis_a']

@pytest.fixture()
def analysis_b(paths):
    return paths['analysis_b']

@pytest.fixture()
def analysis_c(paths):
    return paths['analysis_c']

# session a

@pytest.fixture()
def dg(nwb_a, analysis_a):
    return DriftingGratings.from_analysis_file(BODS(nwb_a), analysis_a)

@pytest.fixture()
def nm1a(nwb_a, analysis_a):
    return NaturalMovie.from_analysis_file(BODS(nwb_a), analysis_a, si.NATURAL_MOVIE_ONE)

@pytest.fixture()
def nm3(nwb_a, analysis_a):
    return NaturalMovie.from_analysis_file(BODS(nwb_a), analysis_a, si.NATURAL_MOVIE_THREE)

# session b

@pytest.fixture()
def sg(nwb_b, analysis_b):
    return StaticGratings.from_analysis_file(BODS(nwb_b), analysis_b)

@pytest.fixture()
def nm1b(nwb_b, analysis_b):
    return NaturalMovie.from_analysis_file(BODS(nwb_b), analysis_b, si.NATURAL_MOVIE_ONE)

@pytest.fixture()
def ns(nwb_b, analysis_b):
    return NaturalScenes.from_analysis_file(BODS(nwb_b), analysis_b)

# session c
@pytest.fixture()
def lsn(nwb_c, analysis_c):
    return LocallySparseNoise.from_analysis_file(BODS(nwb_c), analysis_c, si.LOCALLY_SPARSE_NOISE)

@pytest.fixture()
def nm1c(nwb_c, analysis_c):
    return NaturalMovie.from_analysis_file(BODS(nwb_c), analysis_c, si.NATURAL_MOVIE_ONE)

@pytest.fixture()
def nm2(nwb_c, analysis_c):
    return NaturalMovie.from_analysis_file(BODS(nwb_c), analysis_c, si.NATURAL_MOVIE_TWO)


@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_session_a(dg, nm1a, nm3, nwb_a):
    with tempfile.NamedTemporaryFile(delete=True) as tf:
        save_path = tf.name


    try:
        print("running analysis")
        session_analysis = SessionAnalysis(nwb_a, save_path)
        print(session_analysis.save_dir, save_path)
        session_analysis.session_a(plot_flag=False, save_flag=True)

        print("reading outputs")

        dg_new = DriftingGratings.from_analysis_file(BODS(nwb_a), save_path)
        assert dg.sweep_response.equals(dg_new.sweep_response)
        assert dg.mean_sweep_response.equals(dg_new.mean_sweep_response)
        assert dg.peak.equals(dg_new.peak)
        
        assert np.allclose(dg.response, dg_new.response)
        assert np.allclose(dg.noise_correlation, dg_new.noise_correlation)
        assert np.allclose(dg.signal_correlation, dg_new.signal_correlation)
        assert np.allclose(dg.representational_similarity, dg_new.representational_similarity)

        nm1a_new = NaturalMovie.from_analysis_file(BODS(nwb_a), save_path, si.NATURAL_MOVIE_ONE)
        assert nm1a.sweep_response.equals(nm1a_new.sweep_response)
        assert np.allclose(nm1a.binned_cells_sp, nm1a_new.binned_cells_sp)
        assert np.allclose(nm1a.binned_cells_vis, nm1a_new.binned_cells_vis)
        assert np.allclose(nm1a.binned_dx_sp, nm1a_new.binned_dx_sp)
        assert np.allclose(nm1a.binned_dx_vis, nm1a_new.binned_dx_vis)

        nm3_new = NaturalMovie.from_analysis_file(BODS(nwb_a), save_path, si.NATURAL_MOVIE_THREE)
        assert nm3.sweep_response.equals(nm3_new.sweep_response)
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
        

@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_session_b(sg, nm1b, ns, nwb_b):
    with tempfile.NamedTemporaryFile(delete=True) as tf:
        save_path = tf.name

    try:
        session_analysis = SessionAnalysis(nwb_b, save_path)
        session_analysis.session_b(plot_flag=False, save_flag=True)

        sg_new = StaticGratings.from_analysis_file(BODS(nwb_b), save_path)
        assert sg.sweep_response.equals(sg_new.sweep_response)
        assert sg.mean_sweep_response.equals(sg_new.mean_sweep_response)
        assert sg.peak.equals(sg_new.peak)
    
        assert np.allclose(sg.response, sg_new.response)
        assert np.allclose(sg.noise_correlation, sg_new.noise_correlation)
        assert np.allclose(sg.signal_correlation, sg_new.signal_correlation)
        assert np.allclose(sg.representational_similarity, sg_new.representational_similarity)

        nm1b_new = NaturalMovie.from_analysis_file(BODS(nwb_b), save_path, si.NATURAL_MOVIE_ONE)
        assert nm1b.sweep_response.equals(nm1b_new.sweep_response)

        assert np.allclose(nm1b.binned_cells_sp, nm1b_new.binned_cells_sp)
        assert np.allclose(nm1b.binned_cells_vis, nm1b_new.binned_cells_vis)
        assert np.allclose(nm1b.binned_dx_sp, nm1b_new.binned_dx_sp)
        assert np.allclose(nm1b.binned_dx_vis, nm1b_new.binned_dx_vis)

        ns_new = NaturalScenes.from_analysis_file(BODS(nwb_b), save_path)
        assert ns.sweep_response.equals(ns_new.sweep_response)
        assert ns.mean_sweep_response.equals(ns_new.mean_sweep_response)

        assert np.allclose(ns.noise_correlation, ns_new.noise_correlation)
        assert np.allclose(ns.signal_correlation, ns_new.signal_correlation)
        assert np.allclose(ns.representational_similarity, ns_new.representational_similarity)
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

@pytest.mark.skipif(os.getenv('TEST_COMPLETE') != 'true',
                    reason="partial testing")
def test_session_c(lsn, nm1c, nm2, nwb_c):
    with tempfile.NamedTemporaryFile(delete=True) as tf:
        save_path = tf.name

    try:
        session_analysis = SessionAnalysis(nwb_c, save_path)
        session_analysis.session_c(plot_flag=False, save_flag=True)

        lsn_new = LocallySparseNoise.from_analysis_file(BODS(nwb_c), save_path, si.LOCALLY_SPARSE_NOISE)
        assert lsn.sweep_response.equals(lsn_new.sweep_response)
        assert lsn.mean_sweep_response.equals(lsn_new.mean_sweep_response)
    
        nm1c_new = NaturalMovie.from_analysis_file(BODS(nwb_c), save_path, si.NATURAL_MOVIE_ONE)
        assert nm1c.sweep_response.equals(nm1c_new.sweep_response)

        assert np.allclose(nm1c.binned_dx_sp, nm1c_new.binned_dx_sp) 
        assert np.allclose(nm1c.binned_dx_vis, nm1c_new.binned_dx_vis) 
        assert np.allclose(nm1c.binned_cells_sp, nm1c_new.binned_cells_sp) 
        assert np.allclose(nm1c.binned_cells_vis, nm1c_new.binned_cells_vis) 
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
    
    

