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
