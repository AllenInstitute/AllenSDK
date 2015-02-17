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
import numpy
import re
from json import dump, dumps, loads
from json.encoder import JSONEncoder
from allen_wrench.config.model.configuration_parser import ConfigurationParser
from scipy.sparse import csr_matrix
from numpy import array
from allen_wrench.config.model.description import Description


class NumpyAwareJsonEncoder(JSONEncoder):
    def default(self, o):
        serializable = None
        
        try:
            serializable = super(NumpyAwareJsonEncoder, self).default(o)
        except:
            if isinstance(o, numpy.ndarray):
                serializable = o.tolist()
            elif isinstance(o, csr_matrix):
                serializable = { '__csr__': True,
                                 'data': o.data.tolist(),
                                 'indices': o.indices.tolist(),
                                 'indptr': o.indptr.tolist(),
                                 'shape': o.shape
                                }
        
        return serializable
            
def hinted_hook(obj):
    if '__csr__' in obj:
        data = array(obj['data'])
        indices = array(obj['indices'])
        indptr = array(obj['indptr'])
        shape = (obj['shape'][0], obj['shape'][1])
        return csr_matrix((data, indices, indptr),
                          shape=shape)
    else:
        return obj
            

class JsonConfigurationParser(ConfigurationParser):
    log = logging.getLogger(__name__)
    _oneline_comment_regex = re.compile(r"\/\/.*$",
                                        re.MULTILINE)
    _multiline_comment_regex = re.compile(r"\/\*.*\*\/",
                                          re.MULTILINE | re.DOTALL)
    _blank_line_regex = re.compile(r"\n?^\s*$",
                                   re.MULTILINE)
    _carriage_return_regex = re.compile(r"\r$", re.MULTILINE)
    
    
    def __init__(self):
        super(JsonConfigurationParser, self).__init__()
    
    
    def remove_comments(self, json):
        """ Strip single and multiline javascript-style comments from
            a json string.
            :param json: a json string with javascript-style comments.
            :type json: string
            :return: the json string with comments removed.
            :rtype string:
            
            A json decoder MAY accept and ignore comments.
        """
        json = JsonConfigurationParser._oneline_comment_regex.sub('', json)
        json = JsonConfigurationParser._carriage_return_regex.sub('', json)
        json = JsonConfigurationParser._multiline_comment_regex.sub('', json)
        json = JsonConfigurationParser._blank_line_regex.sub('', json)
        
        return json


    def read(self, file_path, description=None, **kwargs):
        """Read a serialized description from a JSON file.
        
        :parameter filename: the name of the JSON file
        :type filename: string
        :parameter prefix: ignored
        :type prefix: NoneType
        :returns: the description
        :rtype: Description
        """        
        
        try:
            with open(file_path, 'r') as f:
                json_string = f.read()
                description = self.read_string(json_string,
                                               description,
                                               **kwargs)
        except Exception:
            self.log.warn("Couldn't load json description: %s" % file_path)
            raise
        
        return description


    def read_string(self, json_string, description=None, **kwargs):
        if description == None:
            description = Description()

        json_string = self.remove_comments(json_string)
        data = loads(json_string)
        
        unpack_lobs = kwargs.get('unpack_lobs', [])
        self.unpack(description, data, unpack_lobs=unpack_lobs)
        
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
        """Write the configuration to a JSON string.  
        
        :parameter filename: the name of the file to write.
        :type filename: string
        :return json_string: the json serialization of the configuration
        """        
        try:
            json_string = dumps(description.data,
                                indent=2,
                                cls=NumpyAwareJsonEncoder)
            return json_string
        except Exception:
            self.log.warn("Couldn't write allen_wrench json configuration: %s")
            raise
