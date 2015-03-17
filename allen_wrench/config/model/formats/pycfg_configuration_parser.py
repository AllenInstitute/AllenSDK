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

import six
import logging
from pprint import pprint, pformat
from allen_wrench.config.model.description import Description
from allen_wrench.config.model.configuration_parser import ConfigurationParser


class PycfgConfigurationParser(ConfigurationParser):
    log = logging.getLogger(__name__)
    
    def __init__(self):
        super(PycfgConfigurationParser, self).__init__()
    
    
    def read(self, pycfg_file_path, description=None, **kwargs):
        """Read a serialized description from a Python (.pycfg) file.
        
        :parameter filename: the name of the .pycfg file
        :returns the description
        :rtype: Description
        """
        header = kwargs.get('prefix', '')
        
        with open(pycfg_file_path, 'r') as f:
            return self.read_string(f.read(), description, header=header)
    
    
    def read_string(self, python_string, description=None, **kwargs):
        """Read a serialized description from a Python (.pycfg) string.
        
        :parameter python_string: a python string with a serialized description.
        :returns the description object.
        :rtype: Description
        """
        
        if description == None:
            description = Description()
        
        header = kwargs.get('header', '')
                    
        python_string = header + python_string
        
        ns = {}
        code = compile(python_string, 'string', 'exec')
        exec_(code, ns)
        data = ns['allen_wrench_configuration']
        description.unpack(data)
        
        return description


    def write(self, filename, description):
        '''Write the configuration to a Python (.pycfg) file.
        :parameter filename: the name of the file to write.
        :type filename: string
        '''        
        try:
            with open(filename, 'w') as f:
                    pprint(description.data, f, indent=2)

        except Exception:
            self.log.warn("Couldn't write allen_wrench python configuration: %s" % filename)
            raise
        
        return
    
    def write_string(self, description):
        '''Write the configuration to a pretty-printed Python string.
        :parameter description: the configuration object to write
        :type simluation_configuration: SimulationConfiguration
        '''        
        pycfg_string = pformat(description.data, indent=2)
        
        return pycfg_string
        