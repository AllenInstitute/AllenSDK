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


def test_indicator_invalid_reporter_line(monkeypatch):
    """Test that indicator is parsed from full_genotype"""
    class MockExtractor:
        def get_reporter_line(self):
            return 'foo'
    extractor = MockExtractor()

    with monkeypatch.context() as ctx:
        def dummy_init(self):
            self._extractor = extractor

        ctx.setattr(BehaviorOphysMetadata,
                    '__init__',
                    dummy_init)

        metadata = BehaviorOphysMetadata()

        assert metadata.indicator is None
