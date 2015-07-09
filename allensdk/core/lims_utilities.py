import os, platform, re

HDF5_FILE_TYPE_ID = 306905526
NWB_FILE_TYPE_ID = 475137571
METHOD_CONFIG_FILE_TYPE_ID = 324440685
MODEL_PARAMETERS_FILE_TYPE_ID = 329230374
BIOPHYS_MODEL_PARAMETERS_FILE_TYPE_ID = 329230374

def get_well_known_files_by_type(wkfs, wkf_type_id):
    out = [ os.path.join( wkf['storage_directory'], wkf['filename'] )
            for wkf in wkfs
            if wkf.get('well_known_file_type_id',None) == wkf_type_id ]

    if len(out) == 0:
        raise IOError("Could not find well known files with type %d." % wkf_type_id)
    
    return out


def get_well_known_file_by_type(wkfs, wkf_type_id):
    out = get_well_known_files_by_type(wkfs, wkf_type_id)

    nout = len(out)
    if nout != 1:
        raise IOError("Expected single well known file with type %d. Got %d." % (wkf_type_id, nout))
    
    return out[0]


def get_well_known_files_by_name(wkfs, filename):
    out = [ os.path.join( wkf['storage_directory'], wkf['filename'] )
            for wkf in wkfs
            if wkf['filename'] == filename ]

    if len(out) == 0:
        raise IOError("Could not find well known files with name %s." % filename)

    return out

def get_well_known_file_by_name(wkfs, filename):
    out = get_well_known_files_by_name(wkfs, filename)

    nout = len(out)
    if nout != 1:
        raise IOError("Expected single well known file with name %s. Got %d." % (filename, nout))
    
    return out[0]

def append_well_known_file(wkfs, path, wkf_type_id=None, content_type=None):
    record = {
        'filename': os.path.basename(path),
        'storage_directory': os.path.dirname(path)
        }

    if wkf_type_id is not None:
        record['well_known_file_type_id'] = wkf_type_id

    if content_type is not None:
        record['content_type'] = content_type

    for wkf in wkfs:
        if wkf['filename'] == record['filename']:
            wkf.update(record)
            return

    wkfs.append(record)

def connect(user="limsreader", host="limsdb2", database="lims2", password="limsro"):
    import pg8000

    conn = pg8000.connect(user=user, host=host, database=database, password=password)
    return conn.cursor()

def select(cursor, query):
    cursor.execute(query)
    columns = [ d[0] for d in cursor.description ]
    return [ dict(zip(columns, c)) for c in cursor.fetchall() ]

def safe_system_path(file_name):
    if platform.system() == "Windows":
        return linux_to_windows(file_name)
    else:
        return os.path.normpath(file_name)

def linux_to_windows(file_name):
    p = re.compile('/(.*?)/(.*?)/vol1/(.*)')
    m = p.match(file_name)
    
    if m:
        prefix = ""
        if m.group(1) == "data":
            prefix = "\\\\aibsdata"
        elif m.group(1) == "projects":
            prefix = "\\\\titan\\cns"
        
        return os.path.normpath(os.path.join(prefix, m.group(2), m.group(3)))
    else:
        return os.path.normpath(file_name)
