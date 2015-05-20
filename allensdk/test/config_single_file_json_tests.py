import unittest, os
import mock
from mock import MagicMock
from allensdk.model.biophys_sim.config import Config
from allensdk.config.model.formats.json_util import JsonUtil

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
        
        ju = JsonUtil
        ju.read_json_file = MagicMock(return_value=manifest)
    
    
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
    
    real_abspath = os.path.abspath
    
    @mock.patch('allensdk.config.model.manifest.os.path')
    def testGetPathBasedir(self, mock_os_path):
        
        def my_abspath(p):
            if p == 'MOCK_DOT':
                return '/down/this/road'
            else:
                r = ConfigSingleFileJsonTests.real_abspath(p)
                print(r)
                return r
            
        mock_os_path.abspath = MagicMock(side_effect=my_abspath)
        config = Config().load('config.json', False)
        basedir = config.manifest.get_path('BASEDIR')
        mock_os_path.abspath.assert_called_with('MOCK_DOT')
        self.assertEqual('/down/this/road', basedir)
