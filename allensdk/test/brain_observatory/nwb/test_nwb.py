import warnings
import h5py
import pytest

from allensdk.brain_observatory.nwb import check_nwbfile_version


@pytest.fixture
def version_only_nwbfile_fixture(tmp_path, request):

    nwb_version = request.param.get("nwb_version", "2.2.2")

    nwbfile_path = tmp_path / "version_only_nwbfile.nwb"
    with h5py.File(nwbfile_path, "w") as f:
        if nwb_version is not None:
            # pynwb 1.x saves version as a dataset
            # and in the format "NWB-x.y.z"
            if tuple(nwb_version.split(".")) < ("2", "0", "0"):
                f.create_dataset("nwb_version", data=f"NWB-{nwb_version}")
            # pynwb 2.x saves version as an attribute
            elif tuple(nwb_version.split(".")) >= ("2", "0", "0"):
                f.attrs["nwb_version"] = nwb_version
        else:
            f.create_dataset("something_completely_unrelated", data="42")

    return str(nwbfile_path)


@pytest.mark.parametrize("version_only_nwbfile_fixture, min_desired_version"
                         ", warns, warn_msg, invalid_nwb", [
    ({"nwb_version":  None}, "2.2.2" , True, "Warn msg A", True),
    ({"nwb_version": "0.9.0c"}, "2.2.2" , True, "Warn msg B", False),
    ({"nwb_version": "2"}, "2.2.2", True, "Warn msg C", False),
    ({"nwb_version": "2.0b"}, "2.2.2", True, "Warn msg D", False),
    ({"nwb_version": "2.2.2"}, "2.2.2", False, None, False),
    ({"nwb_version": "2.2.8"}, "2.2.2", False, None, False),
    ({"nwb_version": "3.0"}, "2.2.2", False, None, False)
], indirect=["version_only_nwbfile_fixture"])
def test_check_nwbfile_version(version_only_nwbfile_fixture,
                               min_desired_version, warns,
                               warn_msg, invalid_nwb):

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        check_nwbfile_version(nwbfile_path=version_only_nwbfile_fixture,
                              desired_minimum_version=min_desired_version,
                              warning_msg=warn_msg)
    if warns:
        if invalid_nwb:
            assert ("neither a 'nwb_version' field "
                    "nor dataset could be found"
                    in str(w[-1].message))
        else:
            assert warn_msg in str(w[-1].message)
    else:
        assert len(w) == 0
