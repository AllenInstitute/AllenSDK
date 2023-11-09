import pandas as pd
import pytest
from allensdk.brain_observatory.behavior.utils.metadata_parsers import (  # noqa: E501
    parse_behavior_context,
    parse_num_cortical_structures,
    parse_num_depths,
    parse_stimulus_set,
)


def test_parse_behavior_context():
    """Test parsing behavior context for active or passive."""
    df = pd.DataFrame(
        {
            "session_type": [
                "OPHYS_0_images_A_habituation",
                "OPHYS_1_images_A",
                "OPHYS_2_images_A_passive",
                "OPHYS_3_images_A",
                "OPHYS_4_images_A",
                "OPHYS_5_images_A_passive",
                "OPHYS_6_images_A",
            ]
        }
    )
    expected = pd.Series(
        [
            "active_behavior",
            "active_behavior",
            "passive_viewing",
            "active_behavior",
            "active_behavior",
            "passive_viewing",
            "active_behavior",
        ]
    )
    obtained = df["session_type"].apply(parse_behavior_context).rename(None)
    pd.testing.assert_series_equal(expected, obtained)


def test_parse_stimulus_set():
    """Test parsing the stimulus set used in the session from the
    session_type.
    """
    df = pd.DataFrame(
        {
            "session_type": [
                "OPHYS_0_images_A_habituation",
                "OPHYS_1_images_B",
                "OPHYS_2_images_G_passive",
                "TRAINING_1_gratings",
            ]
        }
    )
    expected = pd.Series(
        ["images_A", "images_B", "images_G", "gratings"],
    )
    obtained = df["session_type"].apply(parse_stimulus_set).rename(None)
    pd.testing.assert_series_equal(expected, obtained)

    df = pd.DataFrame(
        {
            "session_type": [
                "OPHYS_0_images_A_habituation",
                "OPHYS_1_images_B",
                "OPHYS_2_images_G_passive",
                "NOT_A_SESSION_TYPE",
                "TRAINING_1_gratings",
            ]
        }
    )
    with pytest.raises(ValueError, match="Session_type NOT_A_SESSION_TYPE"):
        obtained = df["session_type"].apply(parse_stimulus_set)


def test_parse_num_cortical_structures():
    """Test parsing project code into number of structures targeted."""
    df = pd.DataFrame(
        {
            "project_code": [
                "VisualBehavior",
                "VisualBehaviorTask1B",
                "VisualBehaviorMultiscope",
                "VisualBehaviorMultiscope4areasx2d",
                "NotAProject",
            ],
            "ophys_session_id": [0, 1, 2, 3, 4],
        }
    )
    expected = pd.Series([1, 1, 2, 4, None], dtype="Int64")
    obtained = (
        df["project_code"]
        .apply(parse_num_cortical_structures)
        .astype("Int64")
        .rename(None)
    )
    pd.testing.assert_series_equal(expected, obtained)


def test_parse_num_depths():
    """Test parsing the project_code and getting number of depths."""
    df = pd.DataFrame(
        {
            "project_code": [
                "VisualBehavior",
                "VisualBehaviorTask1B",
                "VisualBehaviorMultiscope",
                "VisualBehaviorMultiscope4areasx2d",
                "NotAProject",
            ],
            "ophys_session_id": [0, 1, 2, 3, 4],
        }
    )
    expected = pd.Series([1, 1, 4, 2, None], dtype="Int64")
    obtained = (
        df["project_code"].apply(parse_num_depths).astype("Int64").rename(None)
    )
    pd.testing.assert_series_equal(expected, obtained)
