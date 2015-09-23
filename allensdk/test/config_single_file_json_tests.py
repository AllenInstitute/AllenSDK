import unittest
from mock import MagicMock
from allensdk.model.biophys_sim.config import Config
from allensdk.core.json_utilities import JsonComments

class ConfigSingleFileJsonTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ConfigSingleFileJsonTests, self).__init__(*args, **kwargs)
    
    def setUp(self):
        manifest = {
            'manifest': [
                { 'type': 'dir',
                  'spec': 'MOCK_DOT',
                  'key' : 'BASEDIR'
                }],
            'biophys':
                [{ 'hoc': [ 'stdgui.hoc'] }],
        }

        ju = JsonComments
        ju.read_file = MagicMock(return_value=manifest)
    
    
    def tearDown(self):
        pass
    
    def testAccessHocFilesInData(self):
        config = Config().load('config.json', False)
        self.assertEqual(config.data['biophys'][0]['hoc'][0],
                         'stdgui.hoc')
    
    def testManifestIsNotInData(self):
        config = Config().load('config.json', False)
        self.assertFalse('manifest' in config.data)
    
    
    def testManifestInReservedData(self):
        config = Config().load('config.json', False)
        self.assertTrue('manifest' in config.reserved_data[0])  # huh, why [0]?