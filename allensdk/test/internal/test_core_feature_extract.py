import pytest
from allensdk.internal.ephys.core_feature_extract import ( find_stim_start, 
                                                           filter_sweeps,
                                                           find_coarse_long_square_amp_delta,
                                                           nan_get )

def test_find_stim_start():
    a = [0,0,0,1,1,1,0,0,0]
    idx = find_stim_start(a)
    assert idx == 3

    idx = find_stim_start(a, 1)
    assert idx == 3

    a = [0,0,0,-1,-1,-1,0,0,0]
    idx = find_stim_start(a)
    assert idx == 3

    a = []
    idx = find_stim_start(a)
    assert idx == -1

    a = [0,0,0]
    idx = find_stim_start(a)
    assert idx == -1

    a = [0]
    idx = find_stim_start(a)
    assert idx == -1

def test_filter_sweeps():
    a = [ { 'sweep_number': 1 }, { 'sweep_number': 0 } ]    
    sweeps = filter_sweeps(a, passed_only=False, iclamp_only=False)
    assert len(sweeps) == 2
    assert [ s['sweep_number'] for s in sweeps ] == [ 0,1 ]

    a = [ { 'sweep_number': 1, 'workflow_state': 'auto_passed', 'stimulus_units': 'fish' }, 
          { 'sweep_number': 0,  'workflow_state': 'auto_failed', 'stimulus_units': 'pA' },
          { 'sweep_number': 2, 'workflow_state': 'manual_passed', 'stimulus_units': 'Amps' },
          { 'sweep_number': 3,  'workflow_state': 'manual_failed', 'stimulus_units': 'taco' } ]
    sweeps = filter_sweeps(a, passed_only=True, iclamp_only=False)
    assert len(sweeps) == 2

    sweeps = filter_sweeps(a, passed_only=True, iclamp_only=True)
    assert len(sweeps) == 1

    a = [ { 'sweep_number': 1, 'ephys_stimulus': { 'description': 'T1x' } }, 
          { 'sweep_number': 0, 'ephys_stimulus': { 'description': 'T2x' } },
          { 'sweep_number': 2, 'ephys_stimulus': { 'description': 'T3x' } },
          { 'sweep_number': 3, 'ephys_stimulus': { 'description': 'T1x' } } ]

    sweeps = filter_sweeps(a, passed_only=False, iclamp_only=False, types=['T1', 'T2'])
    assert len(sweeps) == 3

def test_find_coarse_long_square_amp_delta():
    a = [ { 'stimulus_amplitude': 10, 'sweep_number': 1, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' }, 
          { 'stimulus_amplitude': 10, 'sweep_number': 0, 'ephys_stimulus': { 'description': 'C1LSFINE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' },
          { 'stimulus_amplitude': 10, 'sweep_number': 2, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_failed', 'stimulus_units': 'pA' },
          { 'stimulus_amplitude': 10, 'sweep_number': 3, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' } ]

    delta = find_coarse_long_square_amp_delta(a)
    assert delta == 0

    a = [ { 'stimulus_amplitude': 10, 'sweep_number': 1, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' }, 
          { 'stimulus_amplitude': 20, 'sweep_number': 0, 'ephys_stimulus': { 'description': 'C1LSFINE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' },
          { 'stimulus_amplitude': 30, 'sweep_number': 2, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_failed', 'stimulus_units': 'pA' },
          { 'stimulus_amplitude': 40, 'sweep_number': 3, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' } ]

    delta = find_coarse_long_square_amp_delta(a)
    assert delta == 10

    a = [ { 'stimulus_amplitude': 10, 'sweep_number': 1, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' }, 
          { 'stimulus_amplitude': 20, 'sweep_number': 0, 'ephys_stimulus': { 'description': 'C1LSFINE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' },
          { 'stimulus_amplitude': 20, 'sweep_number': 2, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_failed', 'stimulus_units': 'pA' },
          { 'stimulus_amplitude': 30, 'sweep_number': 3, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' } ]

    delta = find_coarse_long_square_amp_delta(a)
    assert delta == 10

    a = [ { 'stimulus_amplitude': 10, 'sweep_number': 0, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' }, 
          { 'stimulus_amplitude': 20, 'sweep_number': 1, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' },
          { 'stimulus_amplitude': 20, 'sweep_number': 2, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_failed', 'stimulus_units': 'pA' },
          { 'stimulus_amplitude': 30, 'sweep_number': 3, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' }, 
          { 'stimulus_amplitude': 50, 'sweep_number': 5, 'ephys_stimulus': { 'description': 'C1LSCOARSE' }, 'workflow_state': 'auto_passed', 'stimulus_units': 'pA' } ]

    delta = find_coarse_long_square_amp_delta(a)
    assert delta == 10

def test_nan_get():
    a = {}
    v = nan_get(a, 'fish')
    assert v == None

    a = { 'fish': 1 }
    v = nan_get(a, 'fish')
    assert v == 1

    a = { 'fish': float("nan") }
    v = nan_get(a, 'fish')
    assert v == None

    

    

    
