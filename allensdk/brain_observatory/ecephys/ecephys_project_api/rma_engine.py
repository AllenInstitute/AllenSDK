import sys

import requests

from .http_engine import HttpEngine


class RmaRequestError(Exception):
    pass


class RmaEngine(HttpEngine):

    @property
    def format_query_string(self):
        return f"query.{self.rma_format}"

    def __init__(self, scheme, host, rma_prefix="api/v2/data", rma_format="json", page_size=5000):
        super(RmaEngine, self).__init__(scheme, host)
        self.rma_prefix = rma_prefix
        self.rma_format = rma_format
        self.page_size = page_size

    def add_page_params(self, url, start, count=None):
        if count is None:
            count = self.page_size
        return f"{url},rma::options[start_row$eq{start}][num_rows$eq{count}]"

    def get(self, query):
        url = f"{self.scheme}://{self.host}/{self.rma_prefix}/{self.format_query_string}?{query}"

        start_row = 0
        total_rows = None

        while total_rows is None or start_row < total_rows:
            current_url = self.add_page_params(url, start_row)
            response_json = requests.get(current_url).json()
            if not response_json["success"]:
                raise RmaRequestError(response_json["msg"])

            start_row += response_json["num_rows"]
            if total_rows is None:
                total_rows = response_json["total_rows"]

            yield response_json["msg"]