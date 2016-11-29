from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
from allensdk.core.nwb_data_set import NwbDataSet

# pick a cell to analyze
specimen_id = 324257146
nwb_file = 'ephys.nwb'

# download the ephys data and sweep metadata
cta = CellTypesApi()
sweeps = cta.get_ephys_sweeps(specimen_id)
cta.save_ephys_data(specimen_id, nwb_file)

# group the sweeps by stimulus 
sweep_numbers = defaultdict(list)
for sweep in sweeps:
    sweep_numbers[sweep['stimulus_name']].append(sweep['sweep_number'])

# calculate features
cell_features = extract_cell_features(NwbDataSet(nwb_file),
                                      sweep_numbers['Ramp'],
                                      sweep_numbers['Short Square'],
                                      sweep_numbers['Long Square'])

