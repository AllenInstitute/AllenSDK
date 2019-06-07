import requests

from .http_engine import HttpEngine


class RmaEngine(HttpEngine):

    @property
    def format_query_string(self):
        return f"query.{self.rma_format}"

    def __init__(self, scheme, host, rma_prefix="api/v2/data", rma_format="json", page_size=5000):
        super(RmaEngine, self).__init__(scheme, host)
        self.rma_prefix = rma_prefix
        self.rma_format = rma_format
        self.page_size = page_size

    def get(self, query):
        url = f"{self.scheme}://{self.host}/{self.rma_prefix}/{self.format_query_string}?{query}"
        return requests.get(url)