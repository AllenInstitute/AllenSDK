import json

import pytest

import numpy as np

from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession


class TestBehaviorEcephysSession:
    @classmethod
    def setup_class(cls):
        with open('/allen/aibs/informatics/module_test_data/ecephys/'
                  'ecephys_session_1111216934_input.json') \
                as f:
            input_data = json.load(f)

        # trim down the number of probes to reduce memory footprint of test
        input_data['session_data']['probes'] = (
                input_data['session_data']['probes'][:3])

        cls._session_from_json = BehaviorEcephysSession.from_json(
            session_data=input_data['session_data']
        )

    @pytest.mark.requires_bamboo
    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        nwbfile = self._session_from_json.to_nwb()

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=nwbfile,
                data_object_cls=BehaviorEcephysSession)
        else:
            obt = BehaviorEcephysSession.from_nwb(nwbfile=nwbfile)

        assert obt == self._session_from_json

    @pytest.mark.requires_bamboo
    def test_session_consistency(self):
        """
        This method will test the self-consistency of
        the BehaviorEcephysSession
        """

        # test that the trials and stimulus_presentations tables
        # agree on the change_frames
        stim = self._session_from_json.stimulus_presentations
        trials = self._session_from_json.trials
        stim_frames = stim[stim.is_change & stim.active].start_frame
        trials_frames = trials[trials.stimulus_change].change_frame
        delta = stim_frames.values-trials_frames.values
        np.testing.assert_array_equal(
            delta,
            np.zeros(len(delta), dtype=int))
