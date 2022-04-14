import datetime
import json

import pytest
from pynwb import NWBFile

from allensdk.brain_observatory.ecephys._behavior_ecephys_metadata import \
    BehaviorEcephysMetadata


class TestBehaviorEcephysMetadata:
    @classmethod
    def setup_class(cls):
        with open('/allen/aibs/informatics/module_test_data/ecephys/'
                  'BEHAVIOR_ECEPHYS_WRITE_NWB_QUEUE_1111216934_input.json') \
                as f:
            input_data = json.load(f)
        input_data = input_data['session_data']
        cls._ecephys_session_id = input_data['ecephys_session_id']
        cls._meta_from_json = BehaviorEcephysMetadata.from_json(
            dict_repr=input_data)

    def setup_method(self, method):
        self._nwbfile = NWBFile(
            session_description='foo',
            identifier=str(self._ecephys_session_id),
            session_id='foo',
            session_start_time=datetime.datetime.now(),
            institution="Allen Institute"
        )

    @pytest.mark.requires_bamboo
    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_read_write_nwb(self, roundtrip,
                            data_object_roundtrip_fixture):
        self._meta_from_json.to_nwb(nwbfile=self._nwbfile)

        if roundtrip:
            obt = data_object_roundtrip_fixture(
                nwbfile=self._nwbfile,
                data_object_cls=BehaviorEcephysMetadata)
        else:
            obt = BehaviorEcephysMetadata.from_nwb(nwbfile=self._nwbfile)

        assert obt == self._meta_from_json
