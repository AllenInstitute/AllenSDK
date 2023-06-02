"""Collection of specific project metadata not readily available on LIMS.
"""


#####################################################################
#
#                   VBO PROJECT CONSTANTS
#
#####################################################################

PROJECT_CODES = [
    "VisualBehavior",
    "VisualBehaviorTask1B",
    "VisualBehaviorMultiscope",
    "VisualBehaviorMultiscope4areasx2d",
]

NUM_STRUCTURES_DICT = {
    "VisualBehavior": 1,
    "VisualBehaviorTask1B": 1,
    "VisualBehaviorMultiscope": 2,
    "VisualBehaviorMultiscope4areasx2d": 4,
}

NUM_DEPTHS_DICT = {
    "VisualBehavior": 1,
    "VisualBehaviorTask1B": 1,
    "VisualBehaviorMultiscope": 4,
    "VisualBehaviorMultiscope4areasx2d": 2,
}

VBO_ACTIVE_MAP = {0: 'initial_gray_screen_5min',
                  1: 'change_detection_behavior',
                  2: 'post_behavior_gray_screen_5min',
                  3: 'natural_movie_one'}

VBO_PASSIVE_MAP = {0: 'initial_gray_screen_5min',
                   1: 'change_detection_passive',
                   2: 'post_behavior_gray_screen_5min',
                   3: 'natural_movie_one'}

VBO_METADATA_COLUMN_ORDER = [
             'behavior_session_id', 'ophys_session_id', 'ophys_container_id',
             'mouse_id', 'indicator', 'full_genotype', 'driver_line',
             'cre_line', 'reporter_line', 'sex', 'age_in_days',
             'imaging_depth', 'targeted_structure', 'targeted_imaging_depth',
             'imaging_plane_group', 'project_code', 'session_type',
             'session_number', 'image_set', 'behavior_type', 'passive',
             'experience_level', 'prior_exposures_to_session_type',
             'prior_exposures_to_image_set', 'prior_exposures_to_omissions',
             'date_of_acquisition', 'equipment_name', 'published_at',
             'isi_experiment_id']
