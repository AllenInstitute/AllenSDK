import json
import tempfile
import pathlib

from allensdk.brain_observatory.behavior.\
    data_files.eye_tracking_metadata_file import (
        EyeTrackingMetadataFile)


def test_eye_tracking_metadata_file(
        tmp_path_factory,
        helper_functions):
    """
    Just a smoke test for EyeTrackingMetadataFile.from_json
    """

    json_path = pathlib.Path(tempfile.mkstemp(suffix='.json')[1])

    data = {'a': [1, {'b': 3}],
            'c': 'x'}

    with open(json_path, 'w') as out_file:
        out_file.write(json.dumps(data))

    dict_repr = {'raw_eye_tracking_video_meta_data':
                 str(json_path.resolve().absolute())}

    data_file = EyeTrackingMetadataFile.from_json(
                    dict_repr=dict_repr)

    assert isinstance(data_file, EyeTrackingMetadataFile)
    assert data_file.data == data

    helper_functions.windows_safe_cleanup(file_path=json_path)
