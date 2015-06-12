import os

HDF5_FILE_TYPE_ID = 306905526
NWB_FILE_TYPE_ID = 475137571

def get_well_known_files_by_type(wkfs, wkf_type_id):
    return [ os.path.join( wkf['storage_directory'], wkf['filename'] )
             for wkf in wkfs
             if wkf['well_known_file_type_id'] == wkf_type_id ]

def get_well_known_files_by_name(wkfs, filename):
    return [ os.path.join( wkf['storage_directory'], wkf['filename'] )
             for wkf in wkfs
             if wkf['filename'] == filename ]


def append_well_known_file(path, wkfs, wkf_type_id):
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)

    for wkf in wkfs:
        if wkf['filename'] == basename:
            wkf['storage_directory'] = dirname
            wkf['well_known_file_type_id'] = wkf_type_id
            return

    wkfs.append({
            'filename': basename,
            'storage_directory': dirname,
            'well_known_file_type_id': wkf_type_id
            })

