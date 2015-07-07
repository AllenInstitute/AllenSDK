# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import json

def read(file_name):
    """ Shortcut reading JSON from a file. """
    with open(file_name, 'rb') as f:
        return json.loads(f.read())

def write(file_name, obj):
    """ Shortcut for writing JSON to a file.  This also takes care of serializing numpy and data types. """
    with open(file_name, 'wb') as f:
        f.write(write_string(obj))

def write_string(obj):
    """ Shortcut for writing JSON to a string.  This also takes care of serializing numpy and data types. """
    return json.dumps(obj, indent=2, default=handler)

def handler(obj):
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
        return long(obj)
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        raise TypeError, 'Object of type %s with value of %s is not JSON serializable' % (type(obj), repr(obj))
