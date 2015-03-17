# Copyright 2014 Allen Institute for Brain Science
# Licensed under the Allen Institute Terms of Use (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.alleninstitute.org/Media/policies/terms_of_use_content.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from allen_wrench.config.model.manifest import Manifest
from allen_wrench.config.model.internal.simulation_configuration import SimulationConfiguration


class Description(SimulationConfiguration):
    _log = logging.getLogger(__name__)
    
    def __init__(self):
        super(Description, self).__init__()
        self.manifest = Manifest()
    
    
    def unpack(self, data):
        self.unpack_manifest(data)
        self.update_data(data)
    
    
    def unpack_manifest(self, data):
        data_manifest = data.pop("manifest", {})
        reserved_data = { "manifest": data_manifest }
        self.reserved_data.append(reserved_data)
        self.manifest.load_config(data_manifest)
    
    
    def fix_unary_sections(self, section_names=None):
        ''' Wrap section contents that don't have the proper
            array surrounding them in an array.
        '''
        if section_names is None:
            section_names = []
        
        for section in section_names:
            if section in self.data:
                if type(self.data[section]) is dict:
                    self.data[section] = [ self.data[section] ];
                    Description._log.warn("wrapped description section %s in an array." % (section))

