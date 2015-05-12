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
from json import loads
from __builtin__ import classmethod


class JsonUtil(object):
    _log = logging.getLogger(__name__)
    _oneline_comment = re.compile(r"\/\/.*$",
                                  re.MULTILINE)
    _multiline_comment_start = re.compile(r"\/\*",
                                          re.MULTILINE |
                                          re.DOTALL)
    _multiline_comment_end = re.compile(r"\*\/",
                                        re.MULTILINE |
                                        re.DOTALL)
    _blank_line = re.compile(r"\n?^\s*$", re.MULTILINE)
    _carriage_return = re.compile(r"\r$", re.MULTILINE)
    
    
    @classmethod
    def read_json_file(cls, json_path):
        '''Transform a JSON contained in a file into an equivalent
        nested python dict.
        
        Parameters
        ----------
        json_path : string
            The input.
        
        Returns
        -------
        dict
            Python version of the input
        
        Note: if the input is a bare array or literal, for example,
        the output will be of the corresponding type.
        '''
        with open(json_path, 'rb') as f:
            json_string = f.read()
        
        return cls.read_json_string(json_string)
    
    
    @classmethod
    def read_json_string(cls, json_string):
        '''Transform a JSON string into an equivalent nested python dict.
        
        Parameters
        ----------
        json_string : string
            The input.
        
        Returns
        -------
        dict
            Python version of the input
        
        Note: if the input is a bare array or literal, for example,
        the output will be of the corresponding type.
        '''
        return loads(cls.remove_comments(json_string))
    
    
    @classmethod
    def remove_comments(cls, json_string):
        '''Strip single and multiline javascript-style comments.
        
        Parameters
        ----------
        json : string
            Json string with javascript-style comments.
            
        Returns
        -------
        string
            Copy of the input with comments removed.
            
        Note: A JSON decoder MAY accept and ignore comments.
        '''
        json_string = JsonUtil._oneline_comment.sub('', json_string)
        json_string = JsonUtil._carriage_return.sub('', json_string)
        json_string = JsonUtil.remove_multiline_comments(json_string)
        json_string = JsonUtil._blank_line.sub('', json_string)
        
        return json_string
    
    
    @classmethod
    def remove_multiline_comments(cls, json_string):
        '''Rebuild input without substrings matching /*...*/.
        
        Parameters
        ----------
        json_string : string
            may or may not contain multiline comments.
        
        Returns
        -------
        string
            Copy of the input without the comments.
        '''
        new_json = []
        start_iter = JsonUtil._multiline_comment_start.finditer(json_string)
        json_slice_start = 0
        
        for comment_start in start_iter:
            json_slice_end = comment_start.start()
            new_json.append(json_string[json_slice_start:json_slice_end])
            search_start = comment_start.end()
            comment_end = JsonUtil._multiline_comment_end.search(json_string[search_start:])
            if comment_end == None:
                break
            else:
                json_slice_start = search_start + comment_end.end()
        new_json.append(json_string[json_slice_start:])
        
        return ''.join(new_json)
