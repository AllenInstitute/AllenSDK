import os
import sys
from allensdk.internal.api.queries.biophysical_module_api \
    import BiophysicalModuleApi
from allensdk.internal.api.queries.biophysical_module_reader \
    import BiophysicalModuleReader

nmr_id = sys.argv[-1]
bma = BiophysicalModuleApi('http://lims2')
result = bma.get_neuronal_model_runs(nmr_id)
r = BiophysicalModuleReader()
r.read_lims_message(result, None)

def well_known_file_path(d):
    return os.path.join(d['storage_directory'], d['filename'])

print(os.path.join(r.neuronal_model_run_dir(),
      'EPHYS_BIOPHYS_SIMULATE_QUEUE_' + nmr_id + '_input.json'))
print(r.stimulus_path())
print(r.fit_parameters_path())
print(r.morphology_path())
print('\n'.join(r.mod_file_paths()))

