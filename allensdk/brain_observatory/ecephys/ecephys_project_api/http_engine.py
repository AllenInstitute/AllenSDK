import logging

import requests

class HttpEngine:
    def __init__(self, scheme, host):
        self.scheme = scheme
        self.host = host

    def stream(self, path):
        url = f"{self.scheme}://{self.host}/{path}"
        print(url)
        
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