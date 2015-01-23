import numpy as np

def read_json(file_name):
    with open(file_name, 'rb') as f:
        return json.loads(f.read())

def write_json(file_name, obj):
    with open(file_name, 'wb') as f:
        f.write(json.dumps(obj, indent=2, default=json_handler))

def json_handler(obj):
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
