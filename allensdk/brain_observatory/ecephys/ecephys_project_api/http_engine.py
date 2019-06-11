import logging

import requests

class HttpEngine:
    def __init__(self, scheme, host):
        self.scheme = scheme
        self.host = host

    def stream(self, path):
        url = f"{self.scheme}://{self.host}/{path}"
        
        response = requests.get(url, stream=True)
        if "Content-length" in response.headers:
            mb = response.headers["Content-length"] / 1024 ** 2
            logging.warning(f"downloading a {mb:3.3}mb file from {url}")

        for chunk in response:            
            yield chunk