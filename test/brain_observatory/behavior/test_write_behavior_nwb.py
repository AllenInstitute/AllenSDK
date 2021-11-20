import mock
from pathlib import Path
import pytest

from allensdk.brain_observatory.behavior.write_behavior_nwb.__main__ import \
    write_behavior_nwb  # noqa: E501


def test_write_behavior_nwb_no_file():
    """
        This function is testing the fail condition of the write_behavior_nwb
        method. The main functionality of the write_behavior_nwb method occurs
        in a try block, and in the case that an exception is raised there is
        functionality in the except block to check if any partial output
        exists, and if so rename that file to have a .error suffix before
        raising the previously mentioned exception.

        This test is checking the case where that partial output does not
        exist. In this case we still want to have the original exception
        returned and avoid a FileNotFound error.

        To ensure that we enter the except block, a value of None is passed
        for the session_data argument. This will cause a TypeError when
        write_behavior_nwb tries to subscript this variable. We are checking
        that, even though no partial output exists, we still get this
        TypeError raised.
    """
    with pytest.raises(TypeError):
        write_behavior_nwb(
            session_data=None,
            nwb_filepath=''
        )


def test_write_behavior_nwb_with_file(tmpdir):
    """
        This function is testing the fail condition of the write_behavior_nwb
        method. The main functionality of the write_behavior_nwb method occurs
        in a try block, and in the case that an exception is raised there is
        functionality in the except block to check if any partial output
        exists, and if so rename that file to have a .error suffix before
        raising the previously mentioned exception.

        This test is checking the case where a partial output file does
        exist. In this case we still want to have the original exception
        returned and avoid a FileNotFound error, but also check that a new
        file with the .error suffix exists.

        To ensure that we enter the except block, a value of None is passed
        for the session_data argument. This will cause a TypeError when
        write_behavior_nwb tries to subscript this variable. To get the
        partial output file to exist, we simply create a Path object and
        call the .touch method.

        This test also patched the os.remove method to do nothing. This is
        necessary because the write_behavior_nwb method checks for any
        existing output and removes it before running.
    """
    # Create the dummy .nwb file
    fake_nwb_fp = Path(tmpdir) / 'fake_nwb.nwb'
    Path(str(fake_nwb_fp) + '.inprogress').touch()

    def mock_os_remove(fp):
        pass

    # Patch the os.remove method to do nothing
    with mock.patch('os.remove', side_effects=mock_os_remove):
        with pytest.raises(TypeError):
            write_behavior_nwb(
                session_data=None,
                nwb_filepath=str(fake_nwb_fp)
            )

            # Check that the new .error file exists, and that we
            # still get the expected exception
            assert Path(str(fake_nwb_fp) + '.error').exists()
