import logging
import functools

import requests
import aiohttp
import nest_asyncio



class HttpEngine:
    def __init__(self, scheme, host, **kwargs):
        self.scheme = scheme
        self.host = host

    def _build_url(self, route):
        return f"{self.scheme}://{self.host}/{route}"

    def stream(self, path):
        url = self._build_url(path)
        
        response = requests.get(url, stream=True)
        response_mb = None
        if "Content-length" in response.headers:
            response_mb = float(response.headers["Content-length"]) / 1024 ** 2

        # TODO: this should be async with write (response.raw.read_chunked?)
        for ii, chunk in enumerate(response):
            if ii == 0:
                size_message = f"{response_mb:3.3}mb" if response_mb is not None else "potentially large"
                logging.warning(f"downloading a {size_message} file from {url}")
            yield chunk


class AsyncHttpEngine(HttpEngine):

    def __init__(self, scheme, host, session=None, **kwargs):
        super(AsyncHttpEngine, self).__init__(scheme, host, **kwargs)
        self.session = session or aiohttp.ClientSession()

    async def _stream_coroutine(self, route, callback):
        url = self._build_url(route)

        async with self.session.get(url) as response:
            await callback(response.content.iter_chunked(1024*10))  # TODO: don't hardcode chunksize

    def stream(self, path):
        return functools.partial(self._stream_coroutine, path)


def write_bytes_from_coroutine(path, coroutine):
    
    import os
    import asyncio
    
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
