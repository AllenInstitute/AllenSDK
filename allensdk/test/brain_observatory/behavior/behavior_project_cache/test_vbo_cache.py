import pytest
import pathlib

from allensdk.brain_observatory.behavior.behavior_project_cache \
    .behavior_project_cache import (
        VisualBehaviorOphysProjectCache)

from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import (
    BehaviorOphysExperiment)


@pytest.mark.requires_bamboo
def test_session_from_vbo_cache():
    """
    Test that a behavior-only session and a behavior-ophys experiment
    can be loaded from a small VBO cache that was saved on-prem.

    This test will catch any changes we make to the code that break
    backwards compatibility with the VBO data release

    This is really just a smoke test
    """
    cache_dir = pathlib.Path('/allen/aibs/informatics/'
                             'module_test_data/vbo_cache/'
                             'visual-behavior-ophys-1.0.1')

    ophys_dir = cache_dir / 'behavior_ophys_experiments'
    beh_dir = cache_dir / 'behavior_sessions'

    BehaviorSession.from_nwb_path(
            nwb_path=beh_dir / 'behavior_session_870987812.nwb')
    BehaviorOphysExperiment.from_nwb_path(
            nwb_path=ophys_dir / 'behavior_ophys_experiment_948507789.nwb')

    # these values are baked-in by the choice we made when releasing the
    # VBO 2021 dataset
    assert (VisualBehaviorOphysProjectCache.BUCKET_NAME
            == 'visual-behavior-ophys-data')
    assert (VisualBehaviorOphysProjectCache.PROJECT_NAME
            == 'visual-behavior-ophys')
