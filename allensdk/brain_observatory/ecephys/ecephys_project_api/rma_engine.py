import sys
import logging
import time
import ast

import requests
import pandas as pd

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
        return f"{url},rma::options[start_row$eq{start}][num_rows$eq{count}][order$eq'id']"

    def get_rma(self, query):
        url = f"{self.scheme}://{self.host}/{self.rma_prefix}/{self.format_query_string}?{query}"
        logging.debug(url)

        start_row = 0
        total_rows = None

        start_time = time.time()
        while total_rows is None or start_row < total_rows:
            current_url = self.add_page_params(url, start_row)
            response_json = requests.get(current_url).json()
            if not response_json["success"]:
                raise RmaRequestError(response_json["msg"])

            start_row += response_json["num_rows"]
            if total_rows is None:
                total_rows = response_json["total_rows"]

            logging.debug(f"downloaded {start_row} of {total_rows} records ({time.time() - start_time:.3f} seconds)")
            yield response_json["msg"]


    def get_rma_list(self, query):
        response = []
        for chunk in self.get_rma(query):
            response.extend(chunk)
        return response

    def get_rma_tabular(self, query, try_infer_dtypes=True):
        response = pd.DataFrame(self.get_rma_list(query))

        if try_infer_dtypes:
            response = infer_column_types(response)

        return response


def infer_column_types(dataframe):
    """ RMA queries often come back with string-typed columns. This utility tries to infer numeric types.
    """

    dataframe = dataframe.copy()

    for colname in dataframe.columns:
        try:
            dataframe[colname] = dataframe[colname].apply(ast.literal_eval)
        except (ValueError, SyntaxError):
            continue
    
    dataframe = dataframe.infer_objects()
    return dataframe