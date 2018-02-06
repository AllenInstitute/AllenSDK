import pytest
import os
import numpy as np
from mock import patch, MagicMock
from allensdk.test_utilities import temp_dir


@pytest.fixture
def mock_request():
    return MagicMock()


@pytest.mark.parametrize("ismount,base_path",[
    (True, os.path.normpath(os.path.join('/', 'dev', 'shm'))),
    (False, os.path.dirname(temp_dir.__file__))
])
@patch("numpy.random.randint", side_effect=([1, 2, 3, 4, 5, 6],
                                            [1, 2, 3, 4, 5, 7]))
@patch("os.listdir", return_value=["allensdk_test_123456"])
@patch("os.makedirs")
def test_tmp_dir(os_makedirs, os_listdir, randint,
                 mock_request, ismount, base_path):
    with patch("os.path.exists", return_value=True):
        with patch("os.path.ismount", return_value=ismount):
            path = temp_dir.temp_dir(mock_request)
    mock_request.addfinalizer.assert_called_once()
    expected_path = os.path.join(base_path, "allensdk_test_123457")
    os_makedirs.assert_called_once_with(expected_path)
    assert path == expected_path
    with patch("shutil.rmtree") as mock_rmtree:
        with patch("os.path.exists", return_value=True):
            with pytest.warns(UserWarning):
                # run the finalizer
                mock_request.addfinalizer.call_args[0][0]()
        mock_rmtree.assert_called_once_with(expected_path)
        mock_rmtree.reset_mock()
        with patch("os.path.exists", return_value=False):
            mock_request.addfinalizer.call_args[0][0]()
        mock_rmtree.assert_called_once_with(expected_path)
