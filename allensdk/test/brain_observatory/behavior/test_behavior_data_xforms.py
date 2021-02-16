import numpy as np
from allensdk.brain_observatory.behavior.session_apis.data_transforms import BehaviorDataTransforms  # noqa: E501


def test_get_stimulus_timestamps(monkeypatch):
    """
    Test that BehaviorDataTransforms.get_stimulus_timestamps()
    just returns the sum of the intervalsms field in the
    behavior stimulus pickle file, padded with a zero at the
    first timestamp.
    """

    expected = np.array([0., 0.0001, 0.0003, 0.0006, 0.001])

    def dummy_init(self):
        pass

    def dummy_stimulus_file(self):
        intervalsms = [0.1, 0.2, 0.3, 0.4]
        data = {}
        data['items'] = {}
        data['items']['behavior'] = {}
        data['items']['behavior']['intervalsms'] = intervalsms
        return data

    with monkeypatch.context() as ctx:
        ctx.setattr(BehaviorDataTransforms,
                    '__init__',
                    dummy_init)

        ctx.setattr(BehaviorDataTransforms,
                    '_behavior_stimulus_file',
                    dummy_stimulus_file)

        xform = BehaviorDataTransforms()
        timestamps = xform.get_stimulus_timestamps()
        np.testing.assert_array_almost_equal(timestamps,
                                             expected,
                                             decimal=10)
