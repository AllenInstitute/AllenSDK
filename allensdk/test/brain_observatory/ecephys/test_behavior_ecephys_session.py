import json

import pytest

from allensdk.brain_observatory.behavior.data_files import BehaviorStimulusFile
from allensdk.brain_observatory.behavior.data_objects import StimulusTimestamps
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession


class TestBehaviorEcephysSession:
    @classmethod
    def setup_class(cls):
        with open('/allen/aibs/informatics/module_test_data/ecephys/'
                  'BEHAVIOR_ECEPHYS_WRITE_NWB_QUEUE_1111216934_input.json') \
                as f:
            input_data = json.load(f)
        # TODO remove passing of stimulus timestamps here, once
        #  https://app.zenhub.com/workspaces/allensdk-10-5c17f74db59cfb36f158db8c/issues/alleninstitute/allensdk/2337   # noqa
        # is addressed
        ts = StimulusTimestamps.from_stimulus_file(
            stimulus_file=BehaviorStimulusFile.from_json(
                dict_repr=input_data['session_data']),
            monitor_delay=input_data['session_data']['monitor_delay'])
        cls._session_from_json = BehaviorEcephysSession.from_json(
            session_data=input_data,
            stimulus_timestamps=ts
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
