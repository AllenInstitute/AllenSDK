import pytest
import tempfile
import json
import pathlib
import numpy as np

from allensdk.brain_observatory.behavior.data_files.\
    eye_tracking_metadata_file import EyeTrackingMetadataFile

from allensdk.brain_observatory.behavior.\
    data_objects.eye_tracking.eye_tracking_table import (
        get_lost_frames)


@pytest.mark.parametrize(
        "input_str, lost_count, expected_array",
        [('', 0, []),
         ('13-19', 1, [12, 13, 14, 15, 16, 17, 18]),
         ('5-7,100-103', 1, [4, 5, 6, 99, 100, 101, 102]),
         ('77', 1, [76]),
         ('3-5,21-25,201-204', 1,
          [2, 3, 4, 20, 21, 22, 23, 24, 200, 201, 202, 203])])
def test_get_lost_frames(
        input_str,
        lost_count,
        expected_array,
        tmp_path_factory,
        helper_functions):
    """
    Test performance of get_lost_frames by constructing an
    example camera metadata json file with records of lost
    frames and running it through the method.
    """

    metadata = {'RecordingReport':
                {'FramesLostCount': lost_count,
                 'LostFrames': [input_str]}}

    tmpdir = pathlib.Path(tmp_path_factory.mktemp('get_lost_frames'))
    json_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmpdir, suffix='.json')[1])
    with open(json_path, 'w') as output_file:
        output_file.write(json.dumps(metadata))

    dict_repr = {'raw_eye_tracking_video_meta_data':
                 str(json_path.resolve().absolute())}

    metadata = EyeTrackingMetadataFile.from_json(
                    dict_repr=dict_repr)

    actual = get_lost_frames(eye_tracking_metadata=metadata)
    np.testing.assert_array_equal(actual, np.array(expected_array))

    helper_functions.windows_safe_cleanup_dir(dir_path=tmpdir)
