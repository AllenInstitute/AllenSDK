# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# Merchantability Or Fitness FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import pytest
from mock import MagicMock, patch, mock_open
from allensdk.api.api import Api
import allensdk.core.json_utilities as ju
from requests.exceptions import HTTPError
from six.moves import builtins
import requests


@pytest.fixture
def api():
    ju.read_url_post = \
        MagicMock(name='read_url_post',
                  return_value={'whatever': True})

    api = Api()

    return api


@pytest.mark.xfail
def test_failed_download(api):
    with pytest.raises(HTTPError) as e_info:
        api.retrieve_file_over_http('http://example.com/yo.jpg',
                                    '/tmp/testfile')

        assert e_info.typename == 'HTTPError'


def test_request_timeout(api):
    def raise_read_timeout(response, path=None):
        raise requests.exceptions.ReadTimeout

    with patch('requests.get', return_value=MagicMock()) as get_mock:
        response_mock = get_mock.return_value
        response_mock.raise_for_status = MagicMock()
        
        with patch(
            'requests_toolbelt.downloadutils.stream.stream_response_to_file',
            MagicMock(name='stream_response_to_file',
                      side_effect=raise_read_timeout)) as stream_mock:
            with patch(builtins.__name__ + '.open',
                       mock_open(),
                       create=True) as open_mock:
                with patch('os.remove', MagicMock()) as os_remove:
                    with pytest.raises(requests.exceptions.ReadTimeout) as e_info:
                        api.retrieve_file_over_http('http://example.com/yo.jpg',
                                                    '/tmp/testfile')

    assert e_info.typename == 'ReadTimeout'
    stream_mock.assert_called_with(response_mock, path=open_mock.return_value)
    get_mock.assert_called_once_with('http://example.com/yo.jpg',
                                     stream=True,
                                     timeout=(9.05, 31.1))
    open_mock.assert_called_once_with('/tmp/testfile', 'wb')
    os_remove.assert_called_once_with('/tmp/testfile')


def test_do_query_post(api):
    api.do_query(lambda *a, **k: 'http://localhost/%s' % (a[0]),
                 lambda d: d,
                 "wow",
                 post=True)

    ju.read_url_post.assert_called_once_with('http://localhost/wow')


def test_do_query_get(api):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'whatever': True})

    api.do_query(lambda *a, **k: 'http://localhost/%s' % (a[0]),
                 lambda d: d,
                 "wow",
                 post=False)

    ju.read_url_get.assert_called_once_with('http://localhost/wow')


def test_load_api_schema(api):
    ju.read_url_get = \
        MagicMock(name='read_url_get',
                  return_value={'whatever': True})

    api.load_api_schema()

    ju.read_url_get.assert_called_once_with(
        'http://api.brain-map.org/api/v2/data/enumerate.json')
