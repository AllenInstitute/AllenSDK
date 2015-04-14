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
