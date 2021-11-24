from allensdk.brain_observatory.behavior.behavior_session import (
    BehaviorSession)


def test_behavior_session_list_data_attributes_and_methods(monkeypatch):
    """Test that data related methods/attributes/properties for
    BehaviorSession are returned properly."""

    def dummy_init(self):
        pass

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorSession, '__init__', dummy_init)
        bs = BehaviorSession()
        obt = bs.list_data_attributes_and_methods()

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

    assert any(expected ^ set(obt)) is False
