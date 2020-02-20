import os
import pytest
import cv2
from allensdk.brain_observatory.behavior.sync import get_sync_data

def number_of_video_frames(video_path):
    video = cv2.VideoCapture(video_path)
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

base_dir='/allen/programs/braintv/production/visualbehavior/prod0/specimen_789992909/ophys_session_819949602/'
eye_tracking_video_path=os.path.join(base_dir, '819949602_video-1.avi')
behaviormon_video_path=os.path.join(base_dir, '819949602_video-0.avi')
sync_path=os.path.join(base_dir, '819949602_sync.h5')

@pytest.mark.requires_bamboo
@pytest.mark.parametrize('path_to_video, path_to_sync_file, sync_key', [
    pytest.param(behaviormon_video_path, sync_path, 'behavior_monitoring'),
    pytest.param(eye_tracking_video_path, sync_path, 'eye_tracking')
])
def test_video_exposure_time_sync(path_to_video, path_to_sync_file, sync_key):
    n_video_frames = number_of_video_frames(path_to_video)
    n_sync_timestamps = get_sync_data(path_to_sync_file)[sync_key].shape[0]
    assert n_video_frames == n_sync_timestamps
