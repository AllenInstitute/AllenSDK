import logging
import functools
import os
import asyncio
import time
import warnings

import requests
import aiohttp
import nest_asyncio


DEFAULT_TIMEOUT = 10 * 60  # seconds
DEFAULT_CHUNKSIZE = 1024 * 10  # bytes


class HttpEngine:
    def __init__(
        self, 
        scheme, 
        host, 
        timeout=DEFAULT_TIMEOUT, 
        chunksize=DEFAULT_CHUNKSIZE, 
        **kwargs
    ):
        self.scheme = scheme
        self.host = host
        self.timeout = timeout
        self.chunksize = chunksize

    def _build_url(self, route):
        return f"{self.scheme}://{self.host}/{route}"

    def stream(self, route):
        url = self._build_url(route)
        
        start_time = time.time()
        response = requests.get(url, stream=True)
        response_mb = None
        if "Content-length" in response.headers:
            response_mb = float(response.headers["Content-length"]) / 1024 ** 2

        for ii, chunk in enumerate(response.iter_content(self.chunksize)):
            if ii == 0:
                size_message = f"{response_mb:3.3}mb" if response_mb is not None else "potentially large"
                logging.warning(f"downloading a {size_message} file from {url}")
            yield chunk

            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise requests.Timeout(f"Download took {elapsed} seconds, but timeout was set to {self.timeout}")


class AsyncHttpEngine(HttpEngine):

    def __init__(
        self, 
        scheme, 
        host, 
        session=None, 
        **kwargs
    ):
        super(AsyncHttpEngine, self).__init__(scheme, host, **kwargs)

        if session:
            self.session = session
            warnings.warn(
                "Recieved preconstructed session, ignoring timeout parameter."
            )
        else:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.client.ClientTimeout(self.timeout)
            )

    async def _stream_coroutine(self, route, callback):
        url = self._build_url(route)

        async with self.session.get(url) as response:
            await callback(response.content.iter_chunked(self.chunksize))

    def stream(self, route):
        return functools.partial(self._stream_coroutine, route)


def write_bytes_from_coroutine(path, coroutine):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    async def callback(file_, iterable):
        async for chunk in iterable:
            file_.write(chunk)
            
    async def wrapper():
        with open(path, "wb") as file_:
            callback_ = functools.partial(callback, file_)
            await coroutine(callback_)

    nest_asyncio.apply()
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(wrapper())


def write_from_stream(path, stream):
    with open(path, "wb") as fil:
        for chunk in stream:
            fil.write(chunk)
