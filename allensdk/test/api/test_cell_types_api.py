import pytest
from mock import patch, mock_open, MagicMock
from allensdk.api.queries.cell_types_api import CellTypesApi

@pytest.fixture
def mock_cells():
    return [
        { 
            'specimen_tags': [],
            'neuron_reconstructions': [],
            'data_sets': [],
            'donor': {
                'transgenic_lines': [],
                'organism': { 'name': CellTypesApi.MOUSE }
                }
            },
        { 
            'specimen_tags': [],
            'neuron_reconstructions': [],
            'data_sets': [ {} ],
            'donor': {
                'transgenic_lines': [ { 'transgenic_line_type_name': 'driver', 'name': 'fish' } ],
                'organism': { 'name': 'fish' }
                }
            },
        { 
            'specimen_tags': [],
            'neuron_reconstructions': [ {} ],
            'data_sets': [],
            'cell_reporter': { 'name': 'bob' },
            'donor': {
                'transgenic_lines': [],
                'organism': { 'name': CellTypesApi.HUMAN }
                }
            },
        ]

def test_list_cells_unmocked():
    ctapi = CellTypesApi()
    # acceptance test
    cells = ctapi.list_cells()

def test_list_cells_mocked(mock_cells):
    ctapi = CellTypesApi()

    ctapi.model_query = MagicMock(return_value=mock_cells)

    cells = ctapi.list_cells()
    assert len(cells) == 3

    cells = ctapi.list_cells(require_reconstruction=True)
    assert len(cells) == 1
    
    cells = ctapi.list_cells(require_morphology=True)
    assert len(cells) == 1

    cells = ctapi.list_cells(reporter_status=['bob'])
    assert len(cells) == 1

    cells = ctapi.list_cells(species=['HOMO SAPIENS'])
    assert len(cells) == 1

    cells = ctapi.list_cells(species=['mus musculus'])
    assert len(cells) == 1


    

