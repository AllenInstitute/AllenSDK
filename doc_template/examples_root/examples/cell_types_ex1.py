from allensdk.api.queries.cell_types_api import CellTypesApi

ct = CellTypesApi()

# a list of dictionaries containing metadata for cells with reconstructions
cells = ct.list_cells_api(require_reconstruction=True)
print(cells[0])

# download the electrophysiology data for one cell
ct.save_ephys_data(cells[0]['id'], 'example.nwb')

# download the reconstruction for the same cell
ct.save_reconstruction(cells[0]['id'], 'example.swc')
