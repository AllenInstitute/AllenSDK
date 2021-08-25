import pytest

from allensdk.brain_observatory.behavior.metadata.behavior_ophys_metadata \
        import BehaviorOphysMetadata


def test_indicator(monkeypatch):
    """Test that indicator is parsed from full_genotype"""

    class MockExtractor:
        def get_reporter_line(self):
            return 'Ai148(TIT2L-GC6f-ICL-tTA2)'

    extractor = MockExtractor()

    with monkeypatch.context() as ctx:
        def dummy_init(self):
            self._extractor = extractor

        ctx.setattr(BehaviorOphysMetadata,
                    '__init__',
                    dummy_init)

        metadata = BehaviorOphysMetadata()

        assert metadata.indicator == 'GCaMP6f'


@pytest.mark.parametrize("input_reporter_line, warning_msg, expected", (
        (None, 'Error parsing reporter line. It is null.', None),
        ('foo', 'Could not parse indicator from reporter because none'
                'of the expected substrings were found in the reporter', None)
)
                         )
def test_indicator_edge_cases(monkeypatch, input_reporter_line, warning_msg,
                              expected):
    """Test indicator parsing edge cases"""

    class MockExtractor:
        def get_reporter_line(self):
            return input_reporter_line

    extractor = MockExtractor()

    with monkeypatch.context() as ctx:
        def dummy_init(self):
            self._extractor = extractor

        ctx.setattr(BehaviorOphysMetadata,
                    '__init__',
                    dummy_init)

        metadata = BehaviorOphysMetadata()

        with pytest.warns(UserWarning) as record:
            indicator = metadata.indicator
        assert indicator is expected
        assert str(record[0].message) == warning_msg
