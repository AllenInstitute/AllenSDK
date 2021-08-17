from pathlib import Path
from typing import Tuple
import json

import pandas as pd
import pytest

from allensdk.api.cloud_cache.utils import file_hash_from_path
from allensdk.api.cloud_cache.cloud_cache import StaticLocalCache


@pytest.fixture
def mounted_s3_dataset_fixture(tmp_path, request) -> Tuple[Path, str, dict]:
    """A fixture which simulates a project s3 bucket that has been mounted
    as a local directory.
    """

    # Get fixture parameters
    project_name = request.param.get("project_name", "test_project_name_1")
    dataset_version = request.param.get("dataset_version", "0.3.0")
    metadata_file_id_column_name = request.param.get(
        "metadata_file_id_column_name", "file_id"
    )
    metadata_files_contents = request.param.get(
        "metadata_files_contents",
        # Each item in list is a tuple of:
        # (metadata_filename, metadata_contents)
        [
            ("metadata_1.csv", {"mouse": [1, 2, 3], "sex": ["F", "F", "M"]}),
            (
                "metadata_2.csv",
                {
                    "experiment": [4, 5, 6],
                    metadata_file_id_column_name: ["data1", "data2", "data3"]
                }
            )
        ]
    )
    data_files_contents = request.param.get(
        "data_files_contents",
        # Each item in list is a tuple of:
        # (data_filename, data_contents)
        [
            ("data_1.nwb", "123456"),
            ("data_2.nwb", "abcdef"),
            ("data_3.nwb", "ghijkl")
        ]
    )

    # Create mock mounted s3 directory structure
    mock_mounted_base_dir = tmp_path / "mounted_remote_data"
    mock_mounted_base_dir.mkdir()
    mock_project_dir = mock_mounted_base_dir / project_name
    mock_project_dir.mkdir()

    # Create metadata files and manifest entries
    mock_metadata_dir = mock_project_dir / "project_metadata"
    mock_metadata_dir.mkdir()

    manifest_meta_entries = dict()
    for meta_fname, meta_contents in metadata_files_contents:
        meta_save_path = mock_metadata_dir / meta_fname
        df_to_save = pd.DataFrame(meta_contents)
        df_to_save.to_csv(str(meta_save_path), index=False)

        manifest_meta_entries[meta_fname.rstrip(".csv")] = {
            "url": (
                f"http://{project_name}.s3.amazonaws.com/{project_name}"
                f"/project_metadata/{meta_fname}"
            ),
            "version_id": "test_placeholder",
            "file_hash": file_hash_from_path(meta_save_path)
        }

    # Create data files and manifest entries
    mock_data_dir = mock_project_dir / "project_data"
    mock_data_dir.mkdir()

    manifest_data_entries = dict()
    for file_fname, file_contents in data_files_contents:
        file_save_path = mock_data_dir / file_fname
        with file_save_path.open('w') as f:
            f.write(file_contents)

        manifest_data_entries[file_fname.rstrip(".nwb")] = {
            "url": (
                f"http://{project_name}.s3.amazonaws.com/{project_name}"
                f"/project_data/{file_fname}"
            ),
            "version_id": "test_placeholder",
            "file_hash": file_hash_from_path(file_save_path)
        }

    # Create manifest dir and manifest
    mock_manifests_dir = mock_project_dir / "manifests"
    mock_manifests_dir.mkdir()
    manifest_fname = f"test_manifest_v{dataset_version}.json"
    manifest_path = mock_manifests_dir / manifest_fname

    manifest_contents = {
        "project_name": project_name,
        "manifest_version": dataset_version,
        "data_pipeline": [
            {
                "name": "AllenSDK",
                "version": "2.11.0",
                "comment": "This is a test entry. NOT REAL."
            }
        ],
        "metadata_file_id_column_name": metadata_file_id_column_name,
        "metadata_files": manifest_meta_entries,
        "data_files": manifest_data_entries
    }

    with manifest_path.open('w') as f:
        json.dump(manifest_contents, f, indent=4)

    expected = {
        "expected_metadata": metadata_files_contents,
        "expected_data": data_files_contents
    }

    return mock_mounted_base_dir, project_name, expected


@pytest.mark.parametrize(
    "mounted_s3_dataset_fixture",
    [
        {"project_name": "visual-behavior-ophys"}
    ],
    indirect=["mounted_s3_dataset_fixture"]
)
def test_static_local_cache_access(mounted_s3_dataset_fixture):
    local_static_cache_dir, proj_name, expected = mounted_s3_dataset_fixture

    cache = StaticLocalCache(local_static_cache_dir, proj_name)
    cache.load_last_manifest()

    for exp_meta_fname, exp_meta_contents in expected["expected_metadata"]:
        exp_df = pd.DataFrame(exp_meta_contents)
        obt_df_path = cache.metadata_path(exp_meta_fname.rstrip(".csv"))
        obt_df = pd.read_csv(obt_df_path["local_path"])
        pd.testing.assert_frame_equal(exp_df, obt_df)

    for exp_data_fname, exp_data_contents in expected["expected_data"]:
        obt_data_path = cache.data_path(exp_data_fname.rstrip(".nwb"))
        with open(obt_data_path["local_path"], "r") as f:
            obt_data = f.read()
        assert exp_data_contents == obt_data


@pytest.mark.parametrize(
    "num_manifests, project_name, create_project_folders, expected",
    [
        (
            2,
            "test_project",
            True,
            ['test_project_manifest_v0.1.0.json']
        ),
        (
            4,
            "test_project_2",
            True,
            ['test_project_2_manifest_v0.3.0.json']
        ),
        # This test case is expected to raise a RuntimeError
        (
            None,  # Not applicable
            "test_project_2",
            False,
            None  # Not applicable
        )
    ]
)
def test_static_local_cache_list_all_manifests(
    tmp_path, num_manifests, project_name, create_project_folders, expected
):
    cache_dir = tmp_path / "cache_dir"
    cache_dir.mkdir()

    if create_project_folders:
        project_dir = cache_dir / project_name
        project_dir.mkdir()

        manifests_dir = project_dir / "manifests"
        manifests_dir.mkdir()

        for n in range(num_manifests):
            manifest_path = (
                manifests_dir / f"{project_name}_manifest_v0.{n}.0.json"
            )
            manifest_path.touch()

        cache = StaticLocalCache(cache_dir, project_name)

        assert cache._manifest_file_names == expected

    else:
        with pytest.raises(
            RuntimeError, match="Expected the provided cache_dir"
        ):
            _ = StaticLocalCache(cache_dir, project_name)
