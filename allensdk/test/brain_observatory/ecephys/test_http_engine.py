import os

import mock
import requests
import pytest

from allensdk.brain_observatory.ecephys.ecephys_project_api import (
    http_engine
)


class MockResponse:

    @property
    def headers(self):
        return {"Content-length": 10 * 1024 ** 2}
    
    def iter_content(self, chunksize):
        for ii in range(5):
            yield f"{ii}_{chunksize}_".encode()


def test_stream():
    engine = http_engine.HttpEngine(
        scheme="http",
        host="api.brain-map.org/api/v2"
    )

    with mock.patch("requests.get", return_value=MockResponse()) as p:

        results = [item for item in engine.stream("fish")]

        p.assert_called_once_with(
            "http://api.brain-map.org/api/v2/fish", stream=True
        )

        assert f"3_{engine.chunksize}_" == results[3].decode()

def test_stream_timeout():
    engine = http_engine.HttpEngine(
        scheme="http",
        host="api.brain-map.org/api/v2",
        timeout=0
    )

    with mock.patch("requests.get", return_value=MockResponse()):
        with pytest.raises(requests.Timeout):
            for item in engine.stream("fish"):
                pass


def test_stream_to_file(tmpdir_factory):

    tmpdir = str(tmpdir_factory.mktemp("stream_test"))
    path = os.path.join(tmpdir, "look_at_this_file")

    engine = http_engine.HttpEngine(
        scheme="http",
        host="api.brain-map.org/api/v2",
        chunksize="hi"
    )

    with mock.patch("requests.get", return_value=MockResponse()) as p:

        stream = engine.stream("fish")
        http_engine.write_from_stream(path, stream)
    
    with open(path, "r") as fil:
        assert "0_hi_1_hi_2_hi_3_hi_4_hi_" == fil.read()


class MockAsyncSession:
    def get(self, url):
        return MockAsyncResponse()


class MockAsyncResponse:

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return self

    def __await__(self):
        return self

    @property
    def content(self):
        return MockAsyncContent()


class MockAsyncContent:

    async def iter_chunked(self, chunksize):
        for ii in range(10):
            yield (f"{ii}".encode())


def test_async_stream_to_file(tmpdir_factory):
    engine = http_engine.AsyncHttpEngine(
        scheme="http",
        host="api.brain.map.org/api/v2",
        session=MockAsyncSession()
    )

    tmpdir = str(tmpdir_factory.mktemp("async_stream_test"))
    path = os.path.join(tmpdir, "one_two.three")

    stream = engine.stream("foo")
    http_engine.write_bytes_from_coroutine(path, stream)

    with open(path, "r") as fil:
        assert "0123456789" ==  fil.read()

