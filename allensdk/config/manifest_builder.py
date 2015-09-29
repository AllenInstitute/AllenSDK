# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import allensdk.core.json_utilities as ju
import logging


class ManifestBuilder(object):
    def __init__(self):
        self._log = logging.getLogger(__name__)
        self.path_info = []
        self.bps_cfg = {}
        self.stimulus_conf = {}
        self.hoc_conf = {}
    
    
    def add_path(self, key, spec,
                 typename='dir',
                 parent_key=None,
                 format=None):
        entry = {
            'key': key,
            'type': typename,
            'spec': spec }
        
        if format != None:
            entry['format'] = format
        
        if parent_key != None:
            entry['parent_key'] = parent_key
            
        self.path_info.append(entry)
    
    
    def write_json_file(self, path):
        with open(path, 'wb') as f:
            f.write(self.write_json_string())
    
    
    def get_config(self):
        wrapper = { "manifest": self.path_info }
        wrapper.update(self.bps_cfg)
        wrapper.update(self.stimulus_conf)
        wrapper.update(self.hoc_conf)
        
        return wrapper

    
    def write_json_string(self):
        config = self.get_config()
        return ju.write_string(config)