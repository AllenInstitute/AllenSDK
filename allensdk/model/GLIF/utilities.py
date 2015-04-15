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


class MissingSweepException( Exception ): 
    """ An exception that can be used to indicate that a sweep required for processing is not available """
    pass

def read_json(file_name):
    """ Shortcut reading JSON from a file. """
    with open(file_name, 'rb') as f:
        return json.loads(f.read())

def write_json(file_name, obj):
    """ Shortcut for writing JSON to a file.  This also takes care of serializing numpy and data types. """
    with open(file_name, 'wb') as f:
        f.write(json.dumps(obj, indent=2, default=json_handler))

def json_handler(obj):
    """ Used by write_json convert a few non-standard types to things that the json package can handle. """
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        return float(obj)
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        raise TypeError, 'Object of type %s with value of %s is not JSON serializable' % (type(obj), repr(obj))
