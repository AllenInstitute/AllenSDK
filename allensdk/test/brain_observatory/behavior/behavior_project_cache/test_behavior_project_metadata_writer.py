import pytest
from pathlib import Path

import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache
from allensdk.brain_observatory.behavior.behavior_project_cache.external \
    .behavior_project_metadata_writer import \
    BehaviorProjectMetadataWriter
from allensdk.test.brain_observatory.behavior.conftest import get_resources_dir


@pytest.mark.requires_bamboo
def test_metadata(tmpdir):
    """tests that the metadata writer can reproduce metadata files

    Notes
    -----
    this test breaks if any `BehaviorNwb` or `BehaviorOphysNwb` in
    LIMS.well_known_files have been added or subtracted to the set
    with `published_at`=release_date since the expected pkl files
    were pushed to github.

    """
    release_date = '2021-03-25'
    bpc = VisualBehaviorOphysProjectCache.from_lims(
        data_release_date=release_date)
    bpmw = BehaviorProjectMetadataWriter(
        behavior_project_cache=bpc,
        out_dir=str(tmpdir),
        project_name='visual-behavior-ophys',
        data_release_date=release_date)
    bpmw.write_metadata()

    expected_path = (Path(get_resources_dir()) /
                     'project_metadata_writer' / 'expected')

    for table_stem in ['behavior_session_table', 'ophys_session_table',
                       'ophys_experiment_table', 'ophys_cells_table']:
        expected = pd.read_pickle(expected_path / f"{table_stem}.pkl")
        # csv->DataFrame->pkl->DataFrame to match what is in github
        obtained = pd.read_csv(tmpdir / f"{table_stem}.csv")
        obtained.to_pickle(tmpdir / "obtained.pkl", protocol=4)
        obtained = pd.read_pickle(tmpdir / "obtained.pkl")
        pd.testing.assert_frame_equal(expected, obtained)
