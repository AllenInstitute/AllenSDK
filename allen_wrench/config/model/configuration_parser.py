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


class ConfigurationParser(object):
    log = logging.getLogger(__name__)
    
    def __init__(self):
        pass
    
    
    def read(self, file_path, description=None, **kwargs):
        self.reader = self.parser_for_extension(file_path)
        return self.reader.read(file_path, description, **kwargs)
    
    
    def read_string(self, data_string, description=None, header=None):
        # TODO: choose an appropriate subclass and call its read function
        # TODO: allow registering different subclasses        
        raise Exception("Not implemented, use a sub class")
    
    
    # TODO: deprecate
    def unpack(self, description, data, **kwargs):
        description.unpack(data, **kwargs)
    
    
    def write(self, filename, description):
        """Write the description to a file.  
        
        :parameter filename: the name of the file to write.
        :type filename: string
        """
        writer = self.parser_for_extension(filename)
        
        writer.write(filename, description)
    
    
    def parser_for_extension(self, filename):
        # Circular imports
        from allen_wrench.config.model.formats.json_configuration_parser import JsonConfigurationParser
        from allen_wrench.config.model.formats.pycfg_configuration_parser import PycfgConfigurationParser
        
        parser = None
        
        if filename.endswith('.json'):
            parser = JsonConfigurationParser()
        elif filename.endswith('.pycfg'):
            parser = PycfgConfigurationParser()
        else:
            raise Exception('could not determine file format')
            
        return parser