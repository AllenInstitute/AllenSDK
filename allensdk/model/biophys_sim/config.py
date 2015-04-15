# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import re
import logging
from pkg_resources import resource_filename
from allensdk.config.app.application_config import ApplicationConfig
from allensdk.config.model.description_parser import DescriptionParser
from allensdk.config.model.description import Description


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
    
    
    def __init__(self):
        super(Config, self).__init__(Config._DEFAULTS, 
                                     name='biophys', 
                                     halp='tools for biophysically detailed modeling at the Allen Institute.',
                                     pydev=True,
                                     default_log_config=Config._DEFAULT_LOG_CONFIG)
        
        
    def load(self, config_path,
             disable_existing_logs=False):
        super(Config, self).load([config_path], disable_existing_logs)
        description = self.read_model_description()
        
        return description
        
        
    def read_model_description(self):
        reader = DescriptionParser()
        description = Description()
        
        Config._log.info("model file: %s" % self.model_file)
        
        # TODO: make space aware w/ regex
        for model_file in self.model_file.split(','):
            Config._log.info("reading model file %s" % (model_file))
            reader.read(model_file, description)
        
        return description
