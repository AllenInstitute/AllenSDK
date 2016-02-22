from allensdk.core.cell_types_cache import CellTypesCache

from allensdk.api.api import Api
Api.default_api_url = 'http://tcelltypes'

ctc = CellTypesCache()

# a list of cell metadata for cells with reconstructions, download if necessary
cells = ctc.get_cells(require_reconstruction=True)

# open the electrophysiology data of one cell, download if necessary
data_set = ctc.get_ephys_data(cells[0]['id'])

# read the reconstruction, download if necessary
reconstruction = ctc.get_reconstruction(cells[0]['id'])
