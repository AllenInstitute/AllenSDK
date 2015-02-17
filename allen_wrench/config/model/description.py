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
from allen_wrench.config.model.lob_parser import LobParser
from allen_wrench.config.model.internal.simulation_configuration import SimulationConfiguration

class Description(SimulationConfiguration):
    log = logging.getLogger(__name__)
    
    def __init__(self):
        super(Description, self).__init__()

        self.manifest = Manifest()
        self._expanded = {}
        self._final = {}
        
        # these are still used by the experimental model builder
        self.metadata_schema = None
        self.archetypes = None
        self.tables = None
        self.parameter_matrices = None
        
        
    def unpack(self, data, unpack_lobs=None):
        if unpack_lobs is None:
            unpack_lobs = []

        data_manifest = data.pop("manifest", {})
        
        reserved_data = { "manifest": data_manifest }
        
        self.reserved_data.append(reserved_data)
        
        self.manifest.load_config(data_manifest)
        self.update_data(data)
        
        LobParser.unpack_lobs(self.manifest,
                              self.data,
                              unpack_lobs)
