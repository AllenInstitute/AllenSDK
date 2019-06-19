from allensdk.internal.api.lims_api import LimsApi

# These tests just confirm that we can access the endpoint, without asserting
# anything about the output
def test_behavior_video_df_endpoint():
    api = LimsApi()
    bvid_df = api.get_behavior_tracking_video_filepath_df()

def test_eye_tracking_video_df_endpoint():
    api = LimsApi()
    etvid_df = api.get_eye_tracking_video_filepath_df()
