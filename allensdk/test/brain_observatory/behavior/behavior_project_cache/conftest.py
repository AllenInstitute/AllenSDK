import io

import pandas as pd
import pytest
import semver
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io.behavior_neuropixels_project_cloud_api import ( # noqa
    VisualBehaviorNeuropixelsProjectCloudApi,
)
from allensdk.brain_observatory.behavior.behavior_project_cache.project_apis.data_io.behavior_project_cloud_api import ( # noqa
    BehaviorProjectCloudApi,
)


@pytest.fixture
def vbo_s3_cloud_cache_data():

    all_versions = {}
    all_versions["data"] = {}
    all_versions["metadata"] = {}

    cmin, cmax = BehaviorProjectCloudApi.MANIFEST_COMPATIBILITY

    min_compat = semver.parse_version_info(cmin)
    versions = []

    version = str(min_compat)
    versions.append(version)
    data = {}
    metadata = {}

    data["ophys_file_1.nwb"] = {"file_id": 1, "data": b"abcde"}

    data["ophys_file_2.nwb"] = {"file_id": 2, "data": b"fghijk"}

    data["behavior_file_3.nwb"] = {"file_id": 3, "data": b"12345"}

    data["behavior_file_4.nwb"] = {"file_id": 4, "data": b"67890"}

    o_session = [
        {
            "ophys_session_id": 111,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "file_id": 1,
        },
        {
            "ophys_session_id": 222,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "file_id": 2,
        },
    ]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["ophys_session_table"] = bytes(buff.read(), "utf-8")

    b_session = [
        {
            "behavior_session_id": 333,
            "file_id": 3,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "species": "mouse",
        },
        {
            "behavior_session_id": 444,
            "file_id": 4,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "species": "mouse",
        },
    ]
    b_session = pd.DataFrame(b_session)
    buff = io.StringIO()
    b_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["behavior_session_table"] = bytes(buff.read(), "utf-8")

    o_session = [
        {
            "ophys_experiment_id": 5111,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "file_id": 1,
        },
        {
            "ophys_experiment_id": 5222,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "file_id": 2,
        },
    ]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["ophys_experiment_table"] = bytes(buff.read(), "utf-8")

    o_cells = {
        "cell_roi_id": {0: 9080884343, 1: 1080884173, 2: 1080883843},
        "cell_specimen_id": {0: 1086496928, 1: 1086496914, 2: 1086496838},
        "ophys_experiment_id": {0: 775614751, 1: 775614751, 2: 775614751},
    }
    o_cells = pd.DataFrame(o_cells)
    buff = io.StringIO()
    o_cells.to_csv(buff, index=False)
    buff.seek(0)

    metadata["ophys_cells_table"] = bytes(buff.read(), "utf-8")

    all_versions["data"][version] = data
    all_versions["metadata"][version] = metadata

    version = str(min_compat.bump_minor())
    versions.append(version)
    data = {}
    metadata = {}

    data["ophys_file_1.nwb"] = {"file_id": 1, "data": b"lmnopqrs"}

    data["ophys_file_2.nwb"] = {"file_id": 2, "data": b"fghijk"}

    data["behavior_file_3.nwb"] = {"file_id": 3, "data": b"12345"}

    data["behavior_file_4.nwb"] = {"file_id": 4, "data": b"67890"}

    data["ophys_file_5.nwb"] = {"file_id": 5, "data": b"98765"}

    o_session = [
        {
            "ophys_session_id": 222,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "file_id": 1,
        },
        {
            "ophys_session_id": 333,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "file_id": 2,
        },
    ]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["ophys_session_table"] = bytes(buff.read(), "utf-8")

    b_session = [
        {
            "behavior_session_id": 777,
            "file_id": 3,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "species": "mouse",
        },
        {
            "behavior_session_id": 888,
            "file_id": 4,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "species": "mouse",
        },
    ]
    b_session = pd.DataFrame(b_session)
    buff = io.StringIO()
    b_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["behavior_session_table"] = bytes(buff.read(), "utf-8")

    o_session = [
        {
            "ophys_experiment_id": 5444,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "file_id": 1,
        },
        {
            "ophys_experiment_id": 5666,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "file_id": 2,
        },
        {
            "ophys_experiment_id": 5777,
            "mouse_id": "1",
            "date_of_acquisition": "2021-01-01",
            "file_id": 5,
        },
    ]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["ophys_experiment_table"] = bytes(buff.read(), "utf-8")

    o_cells = {
        "cell_roi_id": {0: 1080884343, 1: 1080884173, 2: 1080883843},
        "cell_specimen_id": {0: 1086496928, 1: 1086496914, 2: 1086496838},
        "ophys_experiment_id": {0: 775614751, 1: 775614751, 2: 775614751},
    }
    o_cells = pd.DataFrame(o_cells)
    buff = io.StringIO()
    o_cells.to_csv(buff, index=False)
    buff.seek(0)

    metadata["ophys_cells_table"] = bytes(buff.read(), "utf-8")

    all_versions["data"][version] = data
    all_versions["metadata"][version] = metadata

    return all_versions, versions


@pytest.fixture
def vbn_s3_cloud_cache_data():

    all_versions = {}
    all_versions["data"] = {}
    all_versions["metadata"] = {}

    min, max = VisualBehaviorNeuropixelsProjectCloudApi.MANIFEST_COMPATIBILITY

    min_compat = semver.parse_version_info(min)
    versions = []

    version = str(min_compat)
    versions.append(version)
    data = {}
    metadata = {}

    data["ecephys_file_1.nwb"] = {"file_id": 1, "data": b"abcde"}
    data["ecephys_file_2.nwb"] = {"file_id": 2, "data": b"fghijk"}
    data["probe_5111_lfp.nwb"] = {"file_id": 1024123123, "data": b"0x230213"}
    data["probe_5222_lfp.nwb"] = {"file_id": 1024123124, "data": b"980934"}

    e_session = [
        {"ecephys_session_id": 5111, "file_id": 1},
        {"ecephys_session_id": 5112, "file_id": 2},
    ]
    e_session = pd.DataFrame(e_session)
    buff = io.StringIO()
    e_session.to_csv(buff, index=False)
    buff.seek(0)
    metadata["ecephys_sessions"] = bytes(buff.read(), "utf-8")

    b_session = [
        {
            "behavior_session_id": 333,
            "ecephys_session_id": 5111,
            "species": "mouse",
            "file_id": -999,
        },
        {
            "behavior_session_id": 444,
            "ecephys_session_id": 5112,
            "species": "mouse",
            "file_id": -999,
        },
    ]
    b_session = pd.DataFrame(b_session)
    buff = io.StringIO()
    b_session.to_csv(buff, index=False)
    buff.seek(0)
    metadata["behavior_sessions"] = bytes(buff.read(), "utf-8")

    probes = [
        {
            "ecephys_probe_id": 5111,
            "ecephys_session_id": 5111,
            "has_lfp_data": True,
            "name": "probeA",
            "file_id": 1024123123,
        },
        {
            "ecephys_probe_id": 5222,
            "ecephys_session_id": 5112,
            "has_lfp_data": True,
            "name": "probeA",
            "file_id": 1024123124,
        },
    ]

    probes = pd.DataFrame(probes)
    buff = io.StringIO()
    probes.to_csv(buff, index=False)
    buff.seek(0)

    metadata["probes"] = bytes(buff.read(), "utf-8")

    channels = {
        "ecephys_channel_id": {0: 9080884343, 1: 1080884173, 2: 1080883843},
        "ecephys_probe_id": {0: 1086496928, 1: 1086496914, 2: 1086496838},
        "ecephys_session_id": {0: 775614751, 1: 775614751, 2: 775614751},
    }
    channels = pd.DataFrame(channels)
    buff = io.StringIO()
    channels.to_csv(buff, index=False)
    buff.seek(0)
    metadata["channels"] = bytes(buff.read(), "utf-8")

    units = {
        "unit_id": {0: 9080884343, 1: 1080884173, 2: 1080883843},
        "ecephys_channel_id": {0: 1086496928, 1: 1086496914, 2: 1086496838},
        "ecephys_probe_id": {0: 775614751, 1: 775614751, 2: 775614751},
        "ecephys_session_id": {0: 75614751, 1: 75614751, 2: 75614751},
    }
    units = pd.DataFrame(units)
    buff = io.StringIO()
    units.to_csv(buff, index=False)
    buff.seek(0)
    metadata["units"] = bytes(buff.read(), "utf-8")

    all_versions["data"][version] = data
    all_versions["metadata"][version] = metadata

    # new version:
    version = str(min_compat.bump_minor())
    versions.append(version)
    data = {}
    metadata = {}

    data["ecephys_file_1.nwb"] = {"file_id": 1, "data": b"lmnopqrs"}
    data["ecephys_file_2.nwb"] = {"file_id": 2, "data": b"fghijk"}
    data["ecephys_file_3.nwb"] = {"file_id": 3, "data": b"fxxhijk"}
    data["probe_5411_lfp.nwb"] = {"file_id": 1024123125, "data": b"9890"}
    data["probe_5422_lfp.nwb"] = {"file_id": 1024123126, "data": b"ksdiiondiw"}

    e_session = [
        {
            "ecephys_session_id": 222,
            "file_id": 1,
            "abnormal_histology": None,
            "abnormal_activity": None,
        },
        {
            "ecephys_session_id": 333,
            "file_id": 2,
            "abnormal_histology": ["l", "r"],
            "abnormal_activity": None,
        },
        {
            "ecephys_session_id": 444,
            "file_id": 3,
            "abnormal_histology": None,
            "abnormal_activity": [8, 9],
        },
    ]

    e_session = pd.DataFrame(e_session)
    buff = io.StringIO()
    e_session.to_csv(buff, index=False)
    buff.seek(0)
    metadata["ecephys_sessions"] = bytes(buff.read(), "utf-8")

    b_session = [
        {
            "behavior_session_id": 777,
            "ecephys_session_id": 222,
            "species": "mouse",
            "file_id": -999,
        },
        {
            "behavior_session_id": 888,
            "ecephys_session_id": 333,
            "species": "mouse",
            "file_id": -999,
        },
        {
            "behavior_session_id": 999,
            "ecephys_session_id": 444,
            "species": "mouse",
            "file_id": -999,
        },
    ]

    b_session = pd.DataFrame(b_session)
    buff = io.StringIO()
    b_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["behavior_sessions"] = bytes(buff.read(), "utf-8")

    probes = [
        {
            "ecephys_probe_id": 5411,
            "ecephys_session_id": 222,
            "has_lfp_data": True,
            "name": "probeA",
            "file_id": 1024123125,
        },
        {
            "ecephys_probe_id": 5422,
            "ecephys_session_id": 222,
            "has_lfp_data": True,
            "name": "probeB",
            "file_id": 1024123126,
        },
    ]

    probes = pd.DataFrame(probes)
    buff = io.StringIO()
    probes.to_csv(buff, index=False)
    buff.seek(0)
    metadata["probes"] = bytes(buff.read(), "utf-8")

    channels = {
        "ecephys_channel_id": {0: 9080884343, 1: 1080884173, 2: 1080883843},
        "ecephys_probe_id": {0: 1086496928, 1: 1086496914, 2: 1086496838},
        "ecephys_session_id": {0: 775614751, 1: 775614751, 2: 775614751},
    }
    channels = pd.DataFrame(channels)
    buff = io.StringIO()
    channels.to_csv(buff, index=False)
    buff.seek(0)
    metadata["channels"] = bytes(buff.read(), "utf-8")

    units = {
        "unit_id": {0: 9080884343, 1: 1080884173, 2: 1080883843},
        "ecephys_channel_id": {0: 1086496928, 1: 1086496914, 2: 1086496838},
        "ecephys_probe_id": {0: 775614751, 1: 775614751, 2: 775614751},
        "ecephys_session_id": {0: 75614751, 1: 75614751, 2: 75614751},
    }
    units = pd.DataFrame(units)
    buff = io.StringIO()
    units.to_csv(buff, index=False)
    buff.seek(0)
    metadata["units"] = bytes(buff.read(), "utf-8")

    all_versions["data"][version] = data
    all_versions["metadata"][version] = metadata

    return all_versions, versions


@pytest.fixture
def data_update():
    data = {}
    metadata = {}

    data["ophys_file_1.nwb"] = {"file_id": 1, "data": b"11235"}

    data["ophys_file_2.nwb"] = {"file_id": 2, "data": b"8132134"}

    data["behavior_file_3.nwb"] = {"file_id": 3, "data": b"04916"}

    data["behavior_file_4.nwb"] = {"file_id": 4, "data": b"253649"}

    data["ophys_file_5.nwb"] = {"file_id": 5, "data": b"98765"}

    o_session = [
        {"ophys_session_id": 1110, "file_id": 1},
        {"ophys_session_id": 2220, "file_id": 2},
    ]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["ophys_session_table"] = bytes(buff.read(), "utf-8")

    b_session = [
        {"behavior_session_id": 3330, "file_id": 3, "species": "mouse"},
        {"behavior_session_id": 4440, "file_id": 4, "species": "mouse"},
    ]
    b_session = pd.DataFrame(b_session)
    buff = io.StringIO()
    b_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["behavior_session_table"] = bytes(buff.read(), "utf-8")

    o_session = [
        {"ophys_experiment_id": 6111, "file_id": 1},
        {"ophys_experiment_id": 6222, "file_id": 2},
        {"ophys_experiment_id": 63456, "file_id": 5},
    ]

    o_session = pd.DataFrame(o_session)
    buff = io.StringIO()
    o_session.to_csv(buff, index=False)
    buff.seek(0)

    metadata["ophys_experiment_table"] = bytes(buff.read(), "utf-8")

    o_cells = {
        "cell_roi_id": {0: 9080884343, 1: 1080884173, 2: 1080883843},
        "cell_specimen_id": {0: 1086496928, 1: 1086496914, 2: 1086496838},
        "ophys_experiment_id": {0: 775614751, 1: 775614751, 2: 775614751},
    }
    o_cells = pd.DataFrame(o_cells)
    buff = io.StringIO()
    o_cells.to_csv(buff, index=False)
    buff.seek(0)

    metadata["ophys_cells_table"] = bytes(buff.read(), "utf-8")

    return {"data": data, "metadata": metadata}
