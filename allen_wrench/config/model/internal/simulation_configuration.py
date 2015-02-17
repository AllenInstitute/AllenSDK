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
from copy import copy


class SimulationConfiguration(object):
    log = logging.getLogger(__name__)
    
    def __init__(self):
        self.data = {}
        self.reserved_data = []
        
        
    def update_data(self, data):
        self.data.update(data)
        
        
    def copy_to_data(self, more_data):
        """Copy the expanded representation of the network (without archetypes or matrix parameters) 
        from ``self._final`` to ``self.data`` so it can be used by other methods.  TODO:  deprecate.
        """
        section_names = more_data.keys()
        
        for section_name in section_names:
            self.data[section_name] = copy(more_data[section_name])

            
    def is_empty(self):
        if self.data:
            return False
        
        return True