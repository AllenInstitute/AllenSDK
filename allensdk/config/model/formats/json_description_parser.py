# Copyright 2014 Allen Institute for Brain Science
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

import logging
from json import dump, dumps
from allensdk.config.model.description_parser import DescriptionParser
from allensdk.config.model.formats.json_util import JsonUtil,\
    NumpyAwareJsonEncoder
from allensdk.config.model.description import Description


class JsonDescriptionParser(DescriptionParser):
    log = logging.getLogger(__name__)
    
    
    def __init__(self):
        super(JsonDescriptionParser, self).__init__()
    
    
    def read(self, file_path, description=None, **kwargs):
        """Read a serialized description from a JSON file.
        
        :parameter filename: the name of the JSON file
        :type filename: string
        :parameter prefix: ignored
        :type prefix: NoneType
        """
        if description == None:
            description = Description()
            
        data = JsonUtil.read_json_file(file_path)
        description.unpack(data)
        
        return description
    
    
    def read_string(self, json_string, description=None, **kwargs):
        if description == None:
            description = Description()
        
        data = JsonUtil.read_json_string(json_string)
        
        description.unpack(data)
        
        return description
    
    
    def write(self, filename, description):
        """Write the description to a JSON file.  
        
        :parameter filename: the name of the file to write.
        :type filename: string
        """
        try:
            with open(filename, 'w') as f:
                    dump(description.data, f, indent=2, cls=NumpyAwareJsonEncoder)

        except Exception:
            self.log.warn("Couldn't write allensdk json description: %s" % filename)
            raise
        
        return
    
    
    def write_string(self, description):
        """Write the description to a JSON string.  
        
        :parameter filename: the name of the file to write.
        :type filename: string
        :return json_string: the json serialization of the description
        """        
        try:
            json_string = dumps(description.data,
                                indent=2,
                                cls=NumpyAwareJsonEncoder)
            return json_string
        except Exception:
            self.log.warn("Couldn't write allensdk json description: %s")
            raise
