Cell Types API
--------------

The following example will download a list of cells from the Allen Cell Types Database 
that have neuron reconstructions, then download the electrophysiology NWB and morphology SWC file
for one of those cells::

   from allensdk.api.queries.cell_types_api import CellTypesApi

   ct = CellTypesApi()

   cells = ct.list_cells(require_reconstruction=True)
   ct.save_ephys_data(cells[0]['id'], 'example.nwb')
   ct.save_reconstruction(cells[0]['id'], 'example.swc')
