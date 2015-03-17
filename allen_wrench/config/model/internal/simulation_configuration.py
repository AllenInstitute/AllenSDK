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


class SimulationConfiguration(object):
    log = logging.getLogger(__name__)
    
    def __init__(self):
        self.data = {}
        self.reserved_data = []
        
        
    def update_data(self, data):
        self.data.update(data)
        
        
    def is_empty(self):
        if self.data:
            return False
        
        return True