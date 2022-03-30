import pytest
import pathlib
import platform


class HelperFunctions(object):

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


@pytest.fixture(scope='session')
def helper_functions():
    """
    See solution to making helper functions available across
    a pytest module in
    https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di
    """
    return HelperFunctions
