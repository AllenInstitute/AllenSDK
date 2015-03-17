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
from json import dump, dumps
from allen_wrench.config.model.description_parser import DescriptionParser
from allen_wrench.config.model.formats.json_util import JsonUtil,\
    NumpyAwareJsonEncoder
from allen_wrench.config.model.description import Description


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
            self.log.warn("Couldn't write allen_wrench json description: %s" % filename)
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
            self.log.warn("Couldn't write allen_wrench json description: %s")
            raise
