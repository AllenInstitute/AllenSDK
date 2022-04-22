import json

import pytest

from allensdk.brain_observatory.ecephys.behavior_ecephys_session import \
    BehaviorEcephysSession


class TestBehaviorEcephysSession:
    @classmethod
    def setup_class(cls):
        with open('/allen/aibs/informatics/module_test_data/ecephys/'
                  'BEHAVIOR_ECEPHYS_WRITE_NWB_QUEUE_1111216934_input.json') \
                as f:
            input_data = json.load(f)

        # trim down the number of probes to reduce memory footprint of test
        input_data['session_data']['probes'] = (
                input_data['session_data']['probes'][:3])

        cls._session_from_json = BehaviorEcephysSession.from_json(
            session_data=input_data
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
