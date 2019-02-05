import logging
import subprocess as sp
import shutil
import warnings
import copy as cp

import argschema

from allensdk.config.manifest import Manifest
from ._schemas import InputSchema, OutputSchema, available_hashers
from ..argschema_utilities import write_or_print_outputs


def hash_file(path, hasher_cls):
    with open(path, 'rb') as file_obj:
        hasher = hasher_cls(file_obj)
        hasher.update()
        return hasher.digest()


def copy_file_entry(src, dest, use_rsync, make_parent_dirs):

    if make_parent_dirs:
        Manifest.safe_make_parent_dirs(dest)

        if use_rsync:
            sp.check_call(['rsync', '-a', source, dest])
        else:
            if os.path.isdir(source):
                shutil.copytree(source, dest)
            else:
                shutil.copy(source, dest)
        
        logging.info(f"copied from {source} to {dest}")


def raise_or_warn(message, do_raise, typ=None):
    if do_raise == False:
        typ = UserWarning if typ is None else typ
        warnings.warn(message, typ)

    else:
        typ = ValueError if typ is None else typ
        raise typ(message)
    

def compare(source, dest, hasher_cls, raise_if_comparison_fails):
    if os.path.isdir(source) and os.path.isdir(dest):
        compare_directories(source, dest, hasher_cls, raise_if_comparison_fails)
    elif (not os.path.isdir(source)) and (not os.path.isdir(dest)):
        compare_files(source, dest, hasher_cls, raise_if_comparison_fails)
    else:
        raise_or_warn(f"unable to compare files with directories: {source}, {dest}", raise_if_comparison_fails)


def compare_files(source, dest, hasher_cls, raise_if_comparison_fails):
    source_hash = hash_file(source, hasher_cls)
    dest_hash = hash_file(dest, hasher_cls)

    if source_hash != dest_hash:
        raise_or_warn(f"comparison of {source} and {dest} using {hasher_cls.name} failed", raise_if_comparison_fails)

    return source_hash, dest_hash


def compare_directories(source, dest, hasher_cls, raise_if_comparison_fails):
    source_contents = sorted(os.listdir(source))
    dest_contents = sorted(os.listdir(dest))

    if len(source_contents != len(dest_contents)):
        raise_or_warn(
            f"{source} contains {len(source_contents)} items while {dest} contains {len(dest_contents)} items", 
            raise_if_comparison_fails
        )

    for sitem, ditem in zip(source_contents, dest_contents):
        spath = os.path.join(source, sitem)
        dpath = os.path.join(dest, ditem)

        if sitem != ditem:
            raise_or_warn(f"mismatch between {spath} and {dpath}", raise_if_comparison_fails)
            compare(spath, dpath, hasher_cls, raise_if_comparison_fails)
    

def main(files, use_rsync=True, hasher_key=None, raise_if_comparison_fails=True, make_parent_dirs=True):
    hasher_cls = available_hashers[hasher_key]
    output = []

    for file_entry in files:
        record = cp.deepcopy(file_entry)

        copy_file_entry(file_entry['source'], file_entry['destination'], use_rsync, make_parent_dirs)

        if hasher_cls is not None:
            hashes = compare(file_entry['source'], file_entry['dest'], hasher_cls, raise_if_comparison_fails)
            if hashes is not None:
                record['source_hash'] = hashes[0].decode()
                record['destination_hash'] = hashes[1].decode()

        output.append(record)

    return output
    

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(process)s - %(levelname)s - %(message)s')
    
    parser = argschema.ArgSchemaParser(
        schema_type=InputSchema,
        output_schema_type=OutputSchema,
    )
    
    output = main(**parser.args)
    write_or_print_outputs(output, parser)