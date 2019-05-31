from allensdk.api.queries.rma_api import RmaApi
from .ecephys_project_api import EcephysProjectApi

import pandas as pd
import requests


class EcephysProjectWarehouseApi(EcephysProjectApi):
    
    def __init__(self, 
        scheme="http", host="api.brain-map.org", 
        rma_prefix="api/v2/data/query", rma_format="json", 
        **kwargs
    ):
        self.scheme = scheme
        self.host = host
        self.rma_prefix = rma_prefix
        self.rma_format = rma_format

        super(EcephysProjectWarehouseApi, self).__init__(self, **kwargs)

    def get_session_data(self, session_id):
        files = self._get_session_well_known_files(session_ids=[session_id], wkf_types=["EcephysNwb"])
        download_link = files["download_link"].values[0]

        query = f"{self.scheme}://{self.host}{download_link}"
        return requests.get(query, stream=True)

    def _get_session_well_known_files(self, session_ids, wkf_types):

        session_ids = ",".join(str(sid) for sid in session_ids)
        wkf_types = ",".join(f"'{wkf_type_name}'" for wkf_type_name in wkf_types)

        query = (
            f"{self.scheme}://{self.host}/{self.rma_prefix}.{self.rma_format}?"
            "criteria=model::EcephysSession,"
            f"rma::criteria[id$in{session_ids}],"
            f"rma::include,well_known_files(well_known_file_type[name$in{wkf_types}])"
        )
        json = requests.get(query).json()

        files = []
        for session in json["msg"]:
            for wkf in session["well_known_files"]:
                wkf["well_known_file_type"] = wkf["well_known_file_type"]["name"]
                files.append(wkf)

        return pd.DataFrame(files)
