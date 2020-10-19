import pytest

from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import (
    _get_expt_description
)

OPHYS_1_3_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the mouse "
    "brain as the mouse performs a visual change detection task "
    "with a set of natural scenes upon which it has previously been "
    "trained."
)
OPHYS_2_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the "
    "mouse brain as the mouse is shown images from a "
    "change detection task with a set of natural scenes "
    "upon which it has previously been trained, but with "
    "the lick-response sensor withdrawn (passive/open "
    "loop mode)."
)
OPHYS_4_6_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the mouse "
    "brain as the mouse performs a visual change detection task "
    "with a set of natural scenes that are unique from those on "
    "which it was previously trained."
)
OPHYS_5_DESCRIPTION = (
    "2-photon calcium imaging in the visual cortex of the "
    "mouse brain as the mouse is shown images from a "
    "change detection task with a set of natural scenes "
    "that are unique from those on which it was "
    "previously trained, but with the lick-response "
    "sensor withdrawn (passive/open loop mode)."
)

@pytest.mark.parametrize("session_type, expected_description", [
    ("OPHYS_1_images_A", OPHYS_1_3_DESCRIPTION),
    ("OPHYS_2_images_B", OPHYS_2_DESCRIPTION),
    ("OPHYS_3_images_C", OPHYS_1_3_DESCRIPTION),
    ("OPHYS_4_images_D", OPHYS_4_6_DESCRIPTION),
    ("OPHYS_5_images_E", OPHYS_5_DESCRIPTION),
    ("OPHYS_6_images_F", OPHYS_4_6_DESCRIPTION)
])
def test_get_expt_description_with_valid_session_type(session_type,
                                                      expected_description):
    obt = _get_expt_description(session_type)
    assert obt == expected_description

@pytest.mark.parametrize("session_type", [
    ("bogus_session_type"),
    ("stuff"),
    ("OPHYS_7")
])
def test_get_expt_description_raises_with_invalid_session_type(session_type):
    error_msg_match_phrase = r"Encountered an unknown session type*"
    with pytest.raises(RuntimeError, match=error_msg_match_phrase):
        _ = _get_expt_description(session_type)
