import unittest
from mock import MagicMock
from allensdk.api.queries.rma_api import RmaApi
import pandas as pd
import pandas.io.json as pj
import allensdk.core.json_utilities as ju
from allensdk.api.queries.biophysical_api import \
    BiophysicalApi
from allensdk.config.manifest import Manifest
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.model.biophysical.runner import run
import os
import subprocess
from allensdk.model.biophys_sim.config import Config
from allensdk.core.nwb_data_set import NwbDataSet
import numpy


class BiophysicalTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BiophysicalTests, self).__init__(*args, **kwargs)
    
    
    def setUp(self):
        self.mca = MouseConnectivityApi()
    
    
    def tearDown(self):
        self.cache = None
        self.manifest = None
        self.mca = None
    
    @unittest.skip("requires a big NWB file, takes a while.")
    def test_spike_times(self):
        expected = [
            2.937305,   3.16453 ,   3.24271 ,   4.1622  ,   4.24182 ,
            10.0898  ,  10.132545,  10.176095,  10.2361  ,  10.660655,
            10.762125,  10.863465,  10.93833 ,  11.140815,  11.19246 ,
            11.24553 ,  11.696305,  11.812655,  11.90469 ,  12.056755,
            12.15794 ,  12.233905,  12.47577 ,  12.741295,  12.82861 ,
            12.923175,  18.05068 ,  18.139875,  18.17693 ,  18.221485,
            18.24337 ,  18.39981 ,  18.470705,  18.759675,  18.82183 ,
            18.877475,  18.91033 ,  18.941195,  19.050515,  19.12557 ,
            19.15963 ,  19.188655,  19.226205,  19.29813 ,  19.420665,
            19.47627 ,  19.763365,  19.824225,  19.897995,  19.93155 ,
            20.04916 ,  20.11832 ,  20.148755,  20.18004 ,  20.22173 ,
            20.2433  ,  20.40018 ,  20.470915,  20.759715,  20.82156 ,
            20.866465,  20.90807 ,  20.939175]
        
        bp = BiophysicalApi('http://api.brain-map.org')
        bp.cache_stimulus = True # change to False to not download the large stimulus NWB file
        neuronal_model_id = 472451419    # get this from the web site as above
        bp.cache_data(neuronal_model_id, working_directory='neuronal_model')
        cwd = os.path.realpath(os.curdir)
        print(cwd)
        os.chdir(os.path.join(cwd, 'neuronal_model'))
        manifest = ju.read('manifest.json')
        manifest['biophys'][0]['model_file'][0] = 'manifest_51.json'
        manifest['runs'][0]['sweeps'] = [51]
        ju.write('manifest_51.json', manifest)
        subprocess.call(['nrnivmodl', './modfiles'])
        run(Config().load('manifest_51.json'))
        #os.chdir(cwd)
        nwb_out = NwbDataSet('work/386049444.nwb')
        spikes = nwb_out.get_spike_times(51)
        
        numpy.testing.assert_array_almost_equal(spikes, expected)
