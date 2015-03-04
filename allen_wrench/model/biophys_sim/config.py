import re
import logging
from pkg_resources import resource_filename
from allen_wrench.config.app.application_config import ApplicationConfig
from allen_wrench.config.model.configuration_parser import ConfigurationParser
from allen_wrench.config.model.description import Description

class Config(ApplicationConfig):
    _log = logging.getLogger(__name__)
    
    _DEFAULT_LOG_CONFIG = resource_filename(__name__, 'logging.conf')
    
    #: A structure that defines the available configuration parameters.
    #: The default value and help strings may be seen by viewing the source.
    _DEFAULTS = { 
        'workdir': { 'default': 'workdir',
                     'help': 'writable directory where intermediate and output files are written.' },
        'data_dir': { 'default': '',
                      'help': 'writable directory where intermediate and output files are written.' },
        'model_file': { 'default': 'param.json',
                        'help': 'file where the model parameters are set.' },
        'run_file': { 'default': 'param_run.json',
                      'help': 'file where the run flags are set.' },
        'main': { 'default': 'simulation#run',
                  'help' : 'module#function that runs the actual simulation' }
    }
    
    class DataModelConfig(object):
        def __init__(self, gn=None, d=None, dp=[]):
            ''' Helper class for configuring data model servers'''
            self.group_name = gn # for reference
            self.database_name = d
            self.data_model_paths = dp
        
    def __init__(self, unpack_lobs=['positions',
                                    'connections',
                                    'external_inputs']):
        super(Config, self).__init__(Config._DEFAULTS, 
                                     name='biophys', 
                                     halp='tools for biophysically detailed modelling at the Allen Institute.',
                                     pydev=True,
                                     default_log_config=Config._DEFAULT_LOG_CONFIG)
        self.unpack_lobs=unpack_lobs
        
        
    def load(self, config_path,
             disable_existing_logs=False):
        super(Config, self).load([config_path], disable_existing_logs)
        return self.read_model_run_config(config_path)
        
        
    def read_model_run_config(self, application_config_path):
        reader = ConfigurationParser()
        description = Description()
        
        manifest_default_file = resource_filename(__name__,
                                                  'manifest_default.json')
        reader.read(manifest_default_file, description)
        
        # TODO: make space aware w/ regex
        for model_file in self.model_file.split(','):
            reader.read(model_file, description)
        
        if self.run_file != 'param_run.json':
            reader.read(self.run_file,
                        description,
                        unpack_lobs=self.unpack_lobs)
        
        return description
