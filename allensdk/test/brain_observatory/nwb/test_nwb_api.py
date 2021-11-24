import os

import pytest

from allensdk.brain_observatory.nwb.nwb_api import NwbApi


def test_missing_file(tmpdir_factory):
    path = os.path.join(str(tmpdir_factory.mktemp('nwb_api_missing_file_test')), 'foo.nwb')
    with pytest.raises(OSError):
        NwbApi.from_path(path)