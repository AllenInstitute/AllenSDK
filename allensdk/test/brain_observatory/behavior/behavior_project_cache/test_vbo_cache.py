import pytest
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache\
    .behavior_project_cache import \
        VisualBehaviorOphysProjectCache


@pytest.mark.requires_bamboo
def test_session_from_vbo_cache():
    """
    Test that a behavior-only session and a behavior-ophys experiment
    can be loaded from a small VBO cache that was saved on-prem.

    This test will catch any changes we make to the code that break
    backwards compatibility with the VBO data release

    This is really just a smoke test
    """
    cache_dir = '/allen/aibs/informatics/module_test_data/vbo_cache'

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    session = cache.get_behavior_session(behavior_session_id=870987812)
    experiment = cache.get_behavior_ophys_experiment(
                    ophys_experiment_id=951980471)

    assert isinstance(cache.get_behavior_session_table(), pd.DataFrame)
    assert isinstance(cache.get_ophys_session_table(), pd.DataFrame)
    assert isinstance(cache.get_ophys_experiment_table(), pd.DataFrame)
    assert isinstance(cache.get_ophys_cells_table(), pd.DataFrame)
