from allensdk.api.queries.glif_api import GlifApi
import allensdk.core.json_utilities as ju
from find_optimization_sweeps import find_optimization_sweeps

def test_find_optimization_sweeps():
    ga = GlifApi()
    nm = ga.get_neuronal_model(473836744)
    sweeps = ga.get_ephys_sweeps()
    
    opt_sweeps, stim_index, errs = find_optimization_sweeps(sweeps)
