import os
import sys

import pytest

import datetime
import pytz
from pynwb import NWBFile

from allensdk.test_utilities.custom_comparators import WhitespaceStrippedString
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment)


def get_resources_dir():
    behavior_dir = os.path.dirname(__file__)
    return os.path.join(behavior_dir, 'resources')


def pytest_assertrepr_compare(config, op, left, right):
    if isinstance(left, WhitespaceStrippedString) and op == "==":
        if isinstance(right, WhitespaceStrippedString):
            right_compare = right.orig
        else:
            right_compare = right
        return ["Comparing strings with whitespace stripped. ",
                f"{left.orig} != {right_compare}.", "Diff:"] + left.diff


def pytest_ignore_collect(path, config):
    ''' The brain_observatory.ecephys submodule uses
    python 3.6 features that may not be backwards compatible!
    '''

    if sys.version_info < (3, 6):
        return True
    return False


@pytest.fixture()
def behavior_stimuli_data_fixture(request):
    """
    This fixture mimicks the behavior experiment stimuli data logs and
    allows parameterization for testing
    """
    images_set_log = request.param.get("images_set_log", [
        ('Image', 'im065', 5.809, 0)])
    images_draw_log = request.param.get("images_draw_log", [
        ([0] + [1] * 3 + [0] * 3)
    ])
    grating_set_log = request.param.get("grating_set_log", [
        ('Ori', 90, 3.585, 0)
    ])
    grating_draw_log = request.param.get("grating_draw_log", [
        ([0] + [1] * 3 + [0] * 3)
    ])
    omitted_flash_frame_log = request.param.get("omitted_flash_frame_log", {
        "grating_0": []
    })
    grating_phase = request.param.get("grating_phase", None)
    grating_spatial_frequency = request.param.get("grating_spatial_frequency",
                                                  None)

    has_images = request.param.get("has_images", True)
    has_grating = request.param.get("has_grating", True)

    resources_dir = get_resources_dir()

    image_data = {
        "set_log": images_set_log,
        "draw_log": images_draw_log,
        "image_path": os.path.join(resources_dir,
                                   'stimulus_template',
                                   'input',
                                   'test_image_set.pkl')
    }

    grating_data = {
        "set_log": grating_set_log,
        "draw_log": grating_draw_log,
        "phase": grating_phase,
        "sf": grating_spatial_frequency
    }

    data = {
        "items": {
            "behavior": {
                "stimuli": {},
                "omitted_flash_frame_log": omitted_flash_frame_log
            }
        }
    }

    if has_images:
        data["items"]["behavior"]["stimuli"]["images"] = image_data

    if has_grating:
        data["items"]["behavior"]["stimuli"]["grating"] = grating_data

    return data


@pytest.fixture()
def skeletal_nwb_fixture():
    """
    Instantiate an NWB file that has no real data in it
    """

    timezone = pytz.timezone('UTC')
    date = timezone.localize(datetime.datetime.now())

    nwbfile = NWBFile(
            session_description="dummy",
            identifier="00001",
            session_start_time=date,
            file_create_date=date,
            institution="Allen Institute for Brain Science"
        )
    return nwbfile


@pytest.fixture(scope='session')
def behavior_ophys_experiment_fixture():
    """
    A valid BehaviorOphysExperiment instantiated from_lims
    """

    experiment_id = 953443028
    experiment = BehaviorOphysExperiment.from_lims(
        experiment_id,
        load_stimulus_movie=False)
    return experiment
