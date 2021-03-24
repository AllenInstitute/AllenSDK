import logging

from unittest.mock import MagicMock
from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)


class DummyApi(object):
    def __init__(self):
        pass

    def get_method(self):
        """Method docstring"""
        pass

    def get_no_docstring_method(self):
        pass

    def _other_method(self):
        """Other Method docstring"""
        pass


class DummyApiCache(object):
    def cache_clear(self):
        pass


class SimpleBehaviorSession(BehaviorSession):
    """For the purposes of testing, this class overrides the default
    __init__ of the BehaviorSession. The default __init__ uses
    LazyProperties which expect certain api methods to exist that
    DummyApi and DummyApiCache don't have.
    """
    def __init__(self, api):
        self.api = api


class TestBehaviorSession:
    """Tests for BehaviorSession.
       The vast majority of methods in BehaviorSession are simply calling
       functions from the underlying API. The API required for instantiating a
       BehaviorSession is annotated to show that it requires an class that
       inherits from BehaviorBase, it is ensured that those methods exist in
       the API class. These methods should be covered by unit tests on the
       API class and will not be re-tested here.
    """
    @classmethod
    def setup_class(cls):
        cls.behavior_session = SimpleBehaviorSession(api=DummyApi())

    def test_list_api_methods(self):
        expected = [("get_method", "Method docstring"),
                    ("get_no_docstring_method", "")]
        actual = self.behavior_session.list_api_methods()
        assert expected == actual

    def test_cache_clear_raises_warning(self, caplog):
        expected_msg = ("Attempted to clear API cache, but method"
                        " `cache_clear` does not exist on DummyApi")
        self.behavior_session.cache_clear()
        assert caplog.record_tuples == [
            ("BehaviorSession", logging.WARNING, expected_msg)]

    def test_cache_clear_no_warning(self, caplog):
        caplog.clear()
        bs = SimpleBehaviorSession(api=DummyApiCache())
        bs.cache_clear()
        assert len(caplog.record_tuples) == 0


def test_behavior_session_list_data_attributes_and_methods():
    # Test that data related methods/attributes/properties for
    # BehaviorSession are returned properly.

    # This test will need to be updated if:
    # 1. Data being returned by class has changed
    # 2. Inheritance of class has changed
    expected = {
        'behavior_session_id',
        'get_performance_metrics',
        'get_reward_rate',
        'get_rolling_performance_df',
        'licks',
        'metadata',
        'raw_running_speed',
        'rewards',
        'running_speed',
        'stimulus_presentations',
        'stimulus_templates',
        'stimulus_timestamps',
        'task_parameters',
        'trials'
    }

    behavior_session = BehaviorSession(api=MagicMock())
    obt = behavior_session.list_data_attributes_and_methods()

    assert any(expected ^ set(obt)) is False
