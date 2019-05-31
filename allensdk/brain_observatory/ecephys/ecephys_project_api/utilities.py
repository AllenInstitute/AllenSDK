import requests


def stream_well_known_file_from_lims(wkf_id, scheme="http", lims_host="lims2"):
    url = f"{scheme}://{host}/well_known_files/download/{wkf_id}?wkf_id={wkf_id}"
    return requests.get(url, stream=True)