import pytest, os
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
                'organism': { 'name': CellTypesApi.MOUSE },
                'conditions': [ { 'name': 'disease categories - influenza' } ]
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
                'organism': { 'name': CellTypesApi.HUMAN },
                'conditions': [ { 'name': 'disease categories - cheese' } ]
                }
            },
        ]

@pytest.fixture
def cell_types_api():
    endpoint = None
    
    if 'TEST_API_ENDPOINT' in os.environ:
        endpoint = os.environ['TEST_API_ENDPOINT']
        return CellTypesApi(endpoint)
    else:
        return None

@pytest.mark.skipif(cell_types_api() is None, reason='No TEST_API_ENDPOINT set.')
def test_list_cells_unmocked(cell_types_api):
    from allensdk.config import enable_console_log
    enable_console_log()

    # this test will always require the latest warehouse
    cells = cell_types_api.list_cells()

def test_list_cells_mocked(mock_cells):
    ctapi = CellTypesApi()

    ctapi.model_query = MagicMock(return_value=mock_cells)

    cells = ctapi.list_cells()
    assert len(cells) == 3

    flu_cells = [ cell for cell in cells if cell['disease_categories'] == [('influenza')] ]
    assert len(flu_cells) == 1
    
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


    

