import pytest
import pathlib
import platform
import datetime
from pynwb import NWBFile


class HelperFunctions(object):

    @staticmethod
    def create_blank_nwb_file():
        """
        Create and return an empty NWB file
        """
        nwbfile = NWBFile(
            session_description='foo',
            identifier='1',
            session_id='foo',
            session_start_time=datetime.datetime.now(),
            institution="Allen Institute"
        )
        return nwbfile

    @staticmethod
    def windows_safe_cleanup(file_path: pathlib.Path):
        """
        Try to unlink the specified path. If a PermissionError is raised,
        ignore if the system is Windows (this has been observed on our CI
        systems)
        """
        if file_path.exists():
            try:
                file_path.unlink()
            except PermissionError:
                if platform.system() == "Windows":
                    pass
                else:
                    raise

    @staticmethod
    def _first_pass_safe_cleanup_dir(dir_path: pathlib.Path):
        """
        Unlink all of the files in the directory, then remove the directory.
        If a PermissionError is raised, ignore if the system is Windows
        (this has been observed on our CI systems)
        """
        contents_list = [n for n in dir_path.iterdir()]
        for this_path in contents_list:
            if this_path.is_file():
                HelperFunctions.windows_safe_cleanup(file_path=this_path)
            elif this_path.is_dir():
                HelperFunctions.windows_safe_cleanup_dir(
                        dir_path=this_path)
                try:
                    this_path.rmdir()
                except Exception:
                    pass

    @staticmethod
    def windows_safe_cleanup_dir(dir_path: pathlib.Path):
        """
        Unlink all of the files in the directory, then remove the directory.
        If a PermissionError is raised, ignore if the system is Windows
        (this has been observed on our CI systems)
        """
        HelperFunctions._first_pass_safe_cleanup_dir(
                dir_path=dir_path)

        contents_list = [n for n in dir_path.iterdir()]
        for this_path in contents_list:
            if this_path.is_dir():
                try:
                    this_path.rmdir()
                except Exception:
                    raise


@pytest.fixture(scope='session')
def helper_functions():
    """
    See solution to making helper functions available across
    a pytest module in
    https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di
    """
    return HelperFunctions
