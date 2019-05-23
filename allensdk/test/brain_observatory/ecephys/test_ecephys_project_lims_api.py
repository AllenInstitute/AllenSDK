import os

import pytest
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_project_api import ecephys_project_lims_api as epla


def test_clean_wkf_response(tmpdir_factory):

    tmpdir = str(tmpdir_factory.mktemp('some_dirname'))
    basename = 'foo.txt'

    path = os.path.join(tmpdir, basename)
    with open(path, 'w') as file_obj:
        file_obj.write('...')
        
    df = pd.DataFrame({
        'storage_directory': [tmpdir],
        'filename': [basename]
    })
    expected = pd.DataFrame({
        'path': [path]
    })

    obtained = epla.clean_wkf_response(df)

    pd.testing.assert_frame_equal(expected, obtained, check_like=True)