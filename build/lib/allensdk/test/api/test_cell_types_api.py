import pytest, os
from mock import patch, mock_open, MagicMock
from allensdk.api.queries.cell_types_api import CellTypesApi

@pytest.fixture
def mock_cells_api():
    return [
        { 
            'cell_reporter_status': "fish",
            'csl__x': 1,
            'csl__y': 2,
            'csl__z': 3,
            'donor__species': 'taco',
            'specimen__id': 10,
            'specimen__name': 'joe',
            'structure__layer': 'fifteen',
            'structure_parent__id': 2,
            'structure_parent__acronym': 'ASAP',
            'line_name': 'bezier',
            'tag__dendrite_type': 'spikey',
            'tag__apical': 'stumpy',
            'nr__reconstruction_type': 'fancy',
            'donor__disease_state': 'influenza',
            'donor__id': 1,
            'specimen__hemisphere': 'hi',
            'csl__normalized_depth': 1
        },{

            'cell_reporter_status': "nofish",
            'csl__x': 1,
            'csl__y': 2,
            'csl__z': 3,
            'donor__species': 'taco',
            'specimen__id': 10,
            'specimen__name': 'joe',
            'structure__layer': 'fifteen',
            'structure_parent__id': 2,
            'structure_parent__acronym': 'ASAP',
            'line_name': 'bezier',
            'tag__dendrite_type': 'spikey',
            'tag__apical': 'stumpy',
            'nr__reconstruction_type': None,
            'donor__disease_state': None,
            'donor__id': 1,
            'specimen__hemisphere': 'hi',
            'csl__normalized_depth': 1
        }
    ]
    
    
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


@pytest.mark.requires_api_endpoint
def test_list_cells_unmocked(cell_types_api):
    from allensdk.config import enable_console_log
    enable_console_log()

    # this test will always require the latest warehouse
    cells = cell_types_api.list_cells()


def test_list_cells_mocked(mock_cells):
    with patch.object(CellTypesApi, "model_query", return_value=mock_cells):
        ctapi = CellTypesApi()

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

def test_list_cells_api_mocked(mock_cells_api):
    with patch.object(CellTypesApi, "model_query", return_value=mock_cells_api):
        ctapi = CellTypesApi()

        cells = ctapi.list_cells_api()
        assert len(cells) == 2

        fcells = ctapi.filter_cells_api(cells, require_reconstruction=True)
        assert len(fcells) == 1

        fcells = ctapi.filter_cells_api(cells, require_morphology=True)
        assert len(fcells) == 1

        fcells = ctapi.filter_cells_api(cells, species=['taco'])
        assert len(fcells) == 2

        fcells = ctapi.filter_cells_api(cells, reporter_status=['fish'])
        assert len(fcells) == 1
        

    
