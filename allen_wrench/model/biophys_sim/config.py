import re
import logging
from pkg_resources import resource_filename
from allen_wrench.config.app.application_config import ApplicationConfig

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
        
    def __init__(self):
        super(Config, self).__init__(Config._DEFAULTS, 
                                     name='biophys', 
                                     halp='tools for biophysically detailed modelling at the Allen Institute.',
                                     pydev=True,
                                     default_log_config=Config._DEFAULT_LOG_CONFIG)
        
        
    def load(self, config_path, disable_existing_logs=True):
        super(Config, self).load(config_path, disable_existing_logs)