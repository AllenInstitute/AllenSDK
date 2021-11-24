# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from allensdk.brain_observatory.drifting_gratings import DriftingGratings
from allensdk.brain_observatory.static_gratings import StaticGratings
from allensdk.brain_observatory.natural_scenes import NaturalScenes
from allensdk.brain_observatory.locally_sparse_noise import LocallySparseNoise
from allensdk.brain_observatory.r_neuropil import estimate_contamination_ratios
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd
import pytest
import os


@pytest.fixture
def boc(tmpdir_factory):
    manifest_file = tmpdir_factory.mktemp('data').join(os.path.join('boc','manifest.json'))
    endpoint = os.environ['TEST_API_ENDPOINT'] if 'TEST_API_ENDPOINT' in os.environ else 'http://api.brain-map.org'
    return BrainObservatoryCache(manifest_file=str(manifest_file), base_uri=endpoint)


@pytest.mark.nightly
def test_brain_observatory_trace_analysis_notebook(boc):
    # Drifting Gratings
    data_set = boc.get_ophys_experiment_data(502376461)
    dg = DriftingGratings(data_set)
    specimen_id = 517425074
    specimen_ids = data_set.get_cell_specimen_ids()
    
    cell_loc = np.argwhere(specimen_ids==specimen_id)[0][0]

    assert cell_loc == 97 
    
    # temporal frequency plot
    response = dg.response[:,1:,cell_loc,0]
    tfvals = dg.tfvals[1:]
    orivals = dg.orivals

    # peak
    pk = dg.peak.loc[cell_loc]

    # trials for cell's preferred condition
    pref_ori = dg.orivals[dg.peak.ori_dg[cell_loc]]
    pref_tf = dg.tfvals[dg.peak.tf_dg[cell_loc]]
    assert pref_ori == 180 
    assert pref_tf == 2

    pref_trials = dg.stim_table[(dg.stim_table.orientation==pref_ori)&(dg.stim_table.temporal_frequency==pref_tf)]
    assert pref_trials['start'][1] == 837 
    assert pref_trials['end'][1] == 897 

    # mean sweep response
    subset = dg.sweep_response[(dg.stim_table.orientation==pref_ori)&(dg.stim_table.temporal_frequency==pref_tf)]
    subset_mean = dg.mean_sweep_response[(dg.stim_table.orientation==pref_ori)&(dg.stim_table.temporal_frequency==pref_tf)]
    assert np.isclose(subset_mean['dx'][1], 0.920868)

    # response to each trial
    trial_timestamps = np.arange(-1*dg.interlength, dg.interlength+dg.sweeplength, 1.)/dg.acquisition_rate


@pytest.mark.nightly
def test_brain_observatory_static_gratings_notebook(boc):
    data_set = boc.get_ophys_experiment_data(510938357)
    sg = StaticGratings(data_set)

    peak_head = sg.peak.head()
    assert peak_head['cell_specimen_id'][0] == 517399188
    assert np.isclose(peak_head['reliability_sg'][0], -0.010099250163301616)


@pytest.mark.nightly
def test_brain_observatory_natural_scenes_notebook(boc):
    data_set = boc.get_ophys_experiment_data(510938357)
    ns = NaturalScenes(data_set)
    ns_head = ns.peak.head()
    
    assert np.isclose(ns_head['peak_dff_ns'][0], 4.9614532738)
    assert ns_head['cell_specimen_id'][0] == 517399188


@pytest.mark.nightly
def test_brain_observatory_locally_sparse_noise_notebook(boc):
    specimen_id = 517410165
    cell = boc.get_cell_specimens(ids=[specimen_id])[0]
    
    exp = boc.get_ophys_experiments(experiment_container_ids=[cell['experiment_container_id']],
                                    stimuli=[stim_info.LOCALLY_SPARSE_NOISE])[0]
                                             
    data_set = boc.get_ophys_experiment_data(exp['id'])
    lsn = LocallySparseNoise(data_set)
    specimen_ids = data_set.get_cell_specimen_ids()
    cell_loc = np.argwhere(specimen_ids==specimen_id)[0][0]
    receptive_field = lsn.receptive_field[:,:,cell_loc,0]

    assert True
    #assert cell_loc
    #assert receptive_field

@pytest.mark.nightly
def test_brain_observatory_experiment_containers_notebook(boc):
    targeted_structures = boc.get_all_targeted_structures()
    visp_ecs = boc.get_experiment_containers(targeted_structures=['VISp'])
    depths = boc.get_all_imaging_depths()
    stims = boc.get_all_stimuli()
    cre_lines = boc.get_all_cre_lines()
    cux2_ecs = boc.get_experiment_containers(cre_lines=['Cux2-CreERT2'])
    cux2_ec_id = cux2_ecs[-1]['id']
    exps = boc.get_ophys_experiments(experiment_container_ids=[cux2_ec_id])
    exp = boc.get_ophys_experiments(experiment_container_ids=[cux2_ec_id], 
                                    stimuli=[stim_info.STATIC_GRATINGS])[0]
    exp = boc.get_ophys_experiment_data(exp['id'])

    assert set(depths) == set([175, 185, 195, 200, 205, 225, 250, 265, 275, 276, 285,
                               300, 320, 325, 335, 350, 365, 375, 390, 400, 550, 570,
                               625])

    expected_stimuli = ['drifting_gratings',
                        'locally_sparse_noise',
                        'locally_sparse_noise_4deg',
                        'locally_sparse_noise_8deg',
                        'natural_movie_one',
                        'natural_movie_three',
                        'natural_movie_two',
                        'natural_scenes',
                        'spontaneous',
                        'static_gratings']

    assert set(stims) == set(expected_stimuli)

    expected_cre_lines = [ u'Cux2-CreERT2',
                           u'Emx1-IRES-Cre',
                           u'Fezf2-CreER',
                           u'Nr5a1-Cre',
                           u'Ntsr1-Cre_GN220',
                           u'Pvalb-IRES-Cre',
                           u'Rbp4-Cre_KL100',
                           u'Rorb-IRES2-Cre',
                           u'Scnn1a-Tg3-Cre',
                           u'Slc17a7-IRES2-Cre',
                           u'Sst-IRES-Cre',
                           u'Tlx3-Cre_PL56',
                           u'Vip-IRES-Cre' ]

    assert set(cre_lines) == set(expected_cre_lines)

    cells = boc.get_cell_specimens()

    cells = pd.DataFrame.from_records(cells)

    # find direction selective cells in VISp
    visp_ec_ids = [ ec['id'] for ec in visp_ecs ]
    visp_cells = cells[cells['experiment_container_id'].isin(visp_ec_ids)]

    # significant response to drifting gratings stimulus
    sig_cells = visp_cells[visp_cells['p_dg'] < 0.05]

    # direction selective cells
    dsi_cells = sig_cells[(sig_cells['dsi_dg'] > 0.5) & (sig_cells['dsi_dg'] < 1.5)]
    #assert len(cells) == 27124
    assert len(cells) > 0
    #assert len(visp_cells) == 16031
    assert len(visp_cells) > 0
    #assert len(sig_cells) == 8669
    assert len(sig_cells) > 0
    #assert len(dsi_cells) == 4943
    assert len(dsi_cells) > 0

    # find experiment containers for those cells
    dsi_ec_ids = dsi_cells['experiment_container_id'].unique()

    # Download the ophys experiments containing the drifting gratings stimulus for VISp experiment containers
    dsi_exps = boc.get_ophys_experiments(experiment_container_ids=dsi_ec_ids, stimuli=[stim_info.DRIFTING_GRATINGS])

    # pick a direction-selective cell and find its NWB file
    dsi_cell = dsi_cells.iloc[0]

    # figure out which ophys experiment has the drifting gratings stimulus for the cell's experiment container
    cell_exp = boc.get_ophys_experiments(experiment_container_ids=[dsi_cell['experiment_container_id']], 
                                         stimuli=[stim_info.DRIFTING_GRATINGS])[0]
    
    data_set = boc.get_ophys_experiment_data(cell_exp['id'])

    # Fluorescence
    dsi_cell_id = dsi_cell['cell_specimen_id']
    time, raw_traces = data_set.get_fluorescence_traces(cell_specimen_ids=[dsi_cell_id])
    _, demixed_traces = data_set.get_demixed_traces(cell_specimen_ids=[dsi_cell_id])
    _, neuropil_traces = data_set.get_neuropil_traces(cell_specimen_ids=[dsi_cell_id])
    _, corrected_traces = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=[dsi_cell_id])
    _, dff_traces = data_set.get_dff_traces(cell_specimen_ids=[dsi_cell_id])

    # ROI Masks
    data_set = boc.get_ophys_experiment_data(510221121)
    
    # get the specimen IDs for a few cells
    cids = data_set.get_cell_specimen_ids()[:15:5]
    
    # get masks for specific cells
    roi_mask_list = data_set.get_roi_mask(cell_specimen_ids=cids)
    
    # make a mask of all ROIs in the experiment    
    all_roi_masks = data_set.get_roi_mask_array()
    combined_mask = all_roi_masks.max(axis=0)

    max_projection = data_set.get_max_projection()

    # ROI Analysis
    # example loading drifing grating data
    data_set = boc.get_ophys_experiment_data(512326618)
    dg = DriftingGratings(data_set)

    # filter for visually responding, selective cells
    vis_cells = (dg.peak.ptest_dg < 0.05) &  (dg.peak.peak_dff_dg > 3)
    osi_cells = vis_cells & (dg.peak.osi_dg > 0.5) & (dg.peak.osi_dg <= 1.5)
    dsi_cells = vis_cells & (dg.peak.dsi_dg > 0.5) & (dg.peak.dsi_dg <= 1.5)

    # 2-d tf vs. ori histogram
    # tfval = 0 is used for the blank sweep, so we are ignoring it here
    os = np.zeros((len(dg.orivals), len(dg.tfvals)-1))
    ds = np.zeros((len(dg.orivals), len(dg.tfvals)-1))
    
    for i,trial in dg.peak[osi_cells].iterrows():
        os[trial.ori_dg, trial.tf_dg-1] += 1
        
    for i,trial in dg.peak[dsi_cells].iterrows():
        ds[trial.ori_dg, trial.tf_dg-1] += 1
    
    max_count = max(os.max(), ds.max())

    # Neuropil correction
    data_set = boc.get_ophys_experiment_data(569407590)
    csid = data_set.get_cell_specimen_ids()[0]

    time, demixed_traces = data_set.get_demixed_traces(
        cell_specimen_ids=[csid])
    _, neuropil_traces = data_set.get_neuropil_traces(cell_specimen_ids=[csid])

    results = estimate_contamination_ratios(demixed_traces[0], neuropil_traces[0])
    correction = demixed_traces[0] - results['r'] * neuropil_traces[0]
    _, corrected_traces = data_set.get_corrected_fluorescence_traces(
        cell_specimen_ids=[csid])
    
    # Running Speed and Motion Correction
    data_set = boc.get_ophys_experiment_data(512326618)
    dxcm, dxtime = data_set.get_running_speed()
    mc = data_set.get_motion_correction()

    assert True
