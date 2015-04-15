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

import re, logging
from json.encoder import JSONEncoder
from json import loads
import numpy
from scipy.sparse import csr_matrix
from numpy import array

from __builtin__ import classmethod


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


class JsonUtil(object):
    _log = logging.getLogger(__name__)
    _oneline_comment_regex = re.compile(r"\/\/.*$",
                                        re.MULTILINE)
    _multiline_comment_regex = re.compile(r"\/\*.*\*\/",
                                          re.MULTILINE | re.DOTALL)
    _blank_line_regex = re.compile(r"\n?^\s*$",
                                   re.MULTILINE)
    _carriage_return_regex = re.compile(r"\r$", re.MULTILINE)
    
    
    @classmethod
    def read_json_file(cls, json_path):
        with open(json_path, 'rb') as f:
            json_string = f.read()
        
        return cls.read_json_string(json_string)
    
    
    @classmethod
    def read_json_string(cls, json_string):
        return loads(cls.remove_comments(json_string))
    
    
    @classmethod
    def remove_comments(cls, json_string):
        """ Strip single and multiline javascript-style comments from
            a json string.
            :param json: a json string with javascript-style comments.
            :type json: string
            :return: the json string with comments removed.
            :rtype string:
            
            A json decoder MAY accept and ignore comments.
        """
        json_string = JsonUtil._oneline_comment_regex.sub('', json_string)
        json_string = JsonUtil._carriage_return_regex.sub('', json_string)
        json_string = JsonUtil._multiline_comment_regex.sub('', json_string)
        json_string = JsonUtil._blank_line_regex.sub('', json_string)
        
        return json_string