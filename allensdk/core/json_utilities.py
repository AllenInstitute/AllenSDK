# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import json
import re
import urllib.request, urllib.error, urllib.parse, urllib.parse

def read(file_name):
    """ Shortcut reading JSON from a file. """
    with open(file_name, 'rb') as f:
        string = f.read().decode('utf-8')
        return json.loads(string)


def write(file_name, obj):
    """ Shortcut for writing JSON to a file.  This also takes care of serializing numpy and data types. """
    with open(file_name, 'wb') as f:
        f.write(write_string(obj))


def write_string(obj):
    """ Shortcut for writing JSON to a string.  This also takes care of serializing numpy and data types. """
    return json.dumps(obj, indent=2, default=json_handler)


def read_url(url, method='POST'):
    if method == 'GET':
        return read_url_get(url)
    elif method == 'POST':
        return read_url_post(url)
    else:
        raise Exception('Unknown request method: (%s)' % method)


def read_url_get(url):
    '''Transform a JSON contained in a file into an equivalent
    nested python dict.
    
    Parameters
    ----------
    url : string
    where to get the json.
    
    Returns
    -------
    dict
    Python version of the input
    
    Note: if the input is a bare array or literal, for example,
    the output will be of the corresponding type.
    '''
    response = urllib.request.urlopen(url)
    json_string = response.read().decode('utf-8')
    
    return json.loads(json_string)


def read_url_post(url):
    '''Transform a JSON contained in a file into an equivalent
    nested python dict.
    
    Parameters
    ----------
    url : string
    where to get the json.
    
    Returns
    -------
    dict
    Python version of the input
    
    Note: if the input is a bare array or literal, for example,
    the output will be of the corresponding type.
    '''
    urlp = urllib.parse.urlparse(url)
    main_url = urllib.parse.urlunsplit((urlp.scheme, urlp.netloc, urlp.path, '', ''))
    data = json.dumps(dict(urllib.parse.parse_qsl(urlp.query)))

    handler = urllib.request.HTTPHandler()
    opener = urllib.request.build_opener(handler)

    request = urllib.request.Request(main_url, data)
    request.add_header("Content-Type",'application/json')
    request.get_method = lambda: 'POST'
    
    try:
        response = opener.open(request)
    except Exception as e:
        response = e
        
    if response.code == 200:
        json_string = response.read()
    else:
        json_string = response.read()
        
    return json.loads(json_string)


def json_handler(obj):
    """ Used by write_json convert a few non-standard types to things that the json package can handle. """
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif ( isinstance(obj, np.float64) or 
           isinstance(obj, np.float32) or
           isinstance(obj, np.float16) ):
        return float(obj)
    elif ( isinstance(obj, np.int64) or 
           isinstance(obj, np.int32) or 
           isinstance(obj, np.int16) or 
           isinstance(obj, np.int8) or 
           isinstance(obj, np.uint64) or 
           isinstance(obj, np.uint32) or
           isinstance(obj, np.uint16) or
           isinstance(obj, np.uint8) ):
        return int(obj)
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(obj), repr(obj)))


class JsonComments(object):
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
    def read_string(cls, json_string):
        json_string_no_comments = cls.remove_comments(json_string)
        return json.loads(json_string_no_comments)        


    @classmethod
    def read_file(cls, file_name):
        with open(file_name) as f:
            json_string = f.read()
            return cls.read_string(json_string)


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
        json_string = JsonComments._oneline_comment.sub('', json_string)
        json_string = JsonComments._carriage_return.sub('', json_string)
        json_string = JsonComments.remove_multiline_comments(json_string)
        json_string = JsonComments._blank_line.sub('', json_string)
        
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
        start_iter = JsonComments._multiline_comment_start.finditer(json_string)
        json_slice_start = 0
        
        for comment_start in start_iter:
            json_slice_end = comment_start.start()
            new_json.append(json_string[json_slice_start:json_slice_end])
            search_start = comment_start.end()
            comment_end = JsonComments._multiline_comment_end.search(json_string[search_start:])
            if comment_end == None:
                break
            else:
                json_slice_start = search_start + comment_end.end()
        new_json.append(json_string[json_slice_start:])
        
        return ''.join(new_json)
