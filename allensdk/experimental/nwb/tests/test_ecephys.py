import os

import pytest

from allensdk.experimental.nwb.ecephys.write_file_from_lims import build_file as build_ecephys_nwbfile


# @pytest.mark.skipif(True, reason='depends on resolution of https://github.com/NeurodataWithoutBorders/pynwb/issues/728')
@pytest.mark.skipif(os.environ.get('LIMS2_PASSWORD', None) is None, reason='must supply a lims2 password')
@pytest.mark.parametrize('session_data_set_id', [754312389])
def test_ecephys_roundtrip(roundtripper, session_data_set_id):

    comparisons_dir = os.environ.get('ECEPHYS_ROUNDTRIP_COMPARISONS_DIR', None)
    if comparisons_dir is not None:
        comparisons_dir = os.path.join(comparisons_dir, str(session_data_set_id))

    nwbfile = build_ecephys_nwbfile(session_data_set_id)
    roundtripper(nwbfile, 'roundtrip_test_{}.nwb'.format(session_data_set_id), comparisons_dir)