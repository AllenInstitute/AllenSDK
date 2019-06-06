import requests


class HttpEngine:
    def __init__(self, scheme, host):
        self.scheme = scheme
        self.host = host

    def stream(self, path):
        return requests.get(f"{self.scheme}://{self.host}/{path}", stream=True)
