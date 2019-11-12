import logging

from allensdk.brain_observatory.behavior.behavior_data_session import (
    BehaviorDataSession)


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


class TestBehaviorDataSession:
    """Tests for BehaviorDataSession.
       The vast majority of methods in BehaviorDataSession are simply calling
       functions from the underlying API. The API required for instantiating a
       BehaviorDataSession is annotated to show that it requires an class that
       inherits from BehaviorBase, it is ensured that those methods exist in
       the API class. These methods should be covered by unit tests on the
       API class and will not be re-tested here.
    """
    @classmethod
    def setup_class(cls):
        cls.behavior_session = BehaviorDataSession(api=DummyApi())

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
            ("BehaviorOphysSession", logging.WARNING, expected_msg)]

    def test_cache_clear_no_warning(self, caplog):
        caplog.clear()
        bs = BehaviorDataSession(api=DummyApiCache())
        bs.cache_clear()
        assert len(caplog.record_tuples) == 0
