import pytest
import subprocess

def test_args():
    """
    Test for legacy and newest biophysical model simulation calls
    """
    # Legacy all-active simulation call pattern
    args_legacy = subprocess.check_output(['python', '-m', 'allensdk.test.model.check_parser', 'manifest.json'])
    assert 'stub' not in args_legacy.decode('utf-8')

    # Current all-active simulation call pattern
    args_new = subprocess.check_output(['python', '-m', 'allensdk.test.model.check_parser', 'manifest.json', '--axon_type', 'stub'])
    assert 'stub' in args_new.decode('utf-8')
