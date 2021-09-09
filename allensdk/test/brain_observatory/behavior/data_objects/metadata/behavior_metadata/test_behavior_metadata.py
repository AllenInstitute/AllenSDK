import datetime
import pickle
import uuid
from pathlib import Path

import pynwb
import pytest
import pytz
from uuid import UUID

from allensdk.brain_observatory.behavior.data_files import StimulusFile
from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.behavior_session_uuid import \
    BehaviorSessionUUID
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisition
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.equipment import \
    Equipment
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.session_type import \
    SessionType
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .behavior_metadata.stimulus_frame_rate import \
    StimulusFrameRate
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .subject_metadata.age import \
    Age
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .subject_metadata.driver_line import \
    DriverLine
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .subject_metadata.full_genotype import \
    FullGenotype
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .subject_metadata.mouse_id import \
    MouseId
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .subject_metadata.reporter_line import \
    ReporterLine
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .subject_metadata.sex import \
    Sex
from allensdk.brain_observatory.behavior.data_objects.metadata \
    .subject_metadata.subject_metadata import \
    SubjectMetadata
from allensdk.test.brain_observatory.behavior.data_objects.lims_util import \
    LimsTest


class BehaviorMetaTestCase:
    @classmethod
    def setup_class(cls):
        cls.meta = cls._get_meta()

    @staticmethod
    def _get_meta():
        subject_meta = SubjectMetadata(
            sex=Sex(sex='M'),
            age=Age(age=139),
            reporter_line=ReporterLine(reporter_line="Ai93(TITL-GCaMP6f)"),
            full_genotype=FullGenotype(
                full_genotype="Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;"
                              "Ai93(TITL-GCaMP6f)/wt"),
            driver_line=DriverLine(
                driver_line=["Camk2a-tTA", "Slc17a7-IRES2-Cre"]),
            mouse_id=MouseId(mouse_id=416369)

        )
        behavior_meta = BehaviorMetadata(
            subject_metadata=subject_meta,
            behavior_session_id=BehaviorSessionId(behavior_session_id=4242),
            equipment=Equipment(equipment_name='my_device'),
            stimulus_frame_rate=StimulusFrameRate(stimulus_frame_rate=60.0),
            session_type=SessionType(session_type='Unknown'),
            behavior_session_uuid=BehaviorSessionUUID(
                behavior_session_uuid=uuid.uuid4())
        )
        return behavior_meta


class TestLims(LimsTest):
    @pytest.mark.requires_bamboo
    def test_behavior_session_uuid(self):
        behavior_session_id = 823847007
        meta = BehaviorMetadata.from_lims(
            behavior_session_id=BehaviorSessionId(
                behavior_session_id=behavior_session_id),
            lims_db=self.dbconn
        )
        assert meta.behavior_session_uuid == \
               uuid.UUID('394a910e-94c7-4472-9838-5345aff59ed8')


class TestBehaviorMetadata(BehaviorMetaTestCase):
    def test_cre_line(self):
        """Tests that cre_line properly parsed from driver_line"""
        fg = FullGenotype(
            full_genotype='Sst-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt')
        assert fg.parse_cre_line() == 'Sst-IRES-Cre'

    def test_cre_line_bad_full_genotype(self):
        """Test that cre_line is None and no error raised"""
        fg = FullGenotype(full_genotype='foo')

        with pytest.warns(UserWarning) as record:
            cre_line = fg.parse_cre_line(warn=True)
        assert cre_line is None
        assert str(record[0].message) == 'Unable to parse cre_line from ' \
                                         'full_genotype'

    def test_reporter_line(self):
        """Test that reporter line properly parsed from list"""
        reporter_line = ReporterLine.parse(reporter_line=['foo'])
        assert reporter_line == 'foo'

    def test_reporter_line_str(self):
        """Test that reporter line returns itself if str"""
        reporter_line = ReporterLine.parse(reporter_line='foo')
        assert reporter_line == 'foo'

    @pytest.mark.parametrize("input_reporter_line, warning_msg, expected", (
            (('foo', 'bar'), 'More than 1 reporter line. '
                             'Returning the first one', 'foo'),
            (None, 'Error parsing reporter line. It is null.', None),
            ([], 'Error parsing reporter line. The array is empty', None)
    )
                             )
    def test_reporter_edge_cases(self, input_reporter_line, warning_msg,
                                 expected):
        """Test reporter line edge cases"""
        with pytest.warns(UserWarning) as record:
            reporter_line = ReporterLine.parse(
                reporter_line=input_reporter_line,
                warn=True)
        assert reporter_line == expected
        assert str(record[0].message) == warning_msg

    def test_age_in_days(self):
        """Test that age_in_days properly parsed from age"""
        age = Age._age_code_to_days(age='P123')
        assert age == 123

    @pytest.mark.parametrize("input_age, warning_msg, expected", (
            ('unkown', 'Could not parse numeric age from age code '
                       '(age code does not start with "P")', None),
            ('P', 'Could not parse numeric age from age code '
                  '(no numeric values found in age code)', None)
    )
                             )
    def test_age_in_days_edge_cases(self, monkeypatch, input_age, warning_msg,
                                    expected):
        """Test age in days edge cases"""
        with pytest.warns(UserWarning) as record:
            age_in_days = Age._age_code_to_days(age=input_age, warn=True)

        assert age_in_days is None
        assert str(record[0].message) == warning_msg

    @pytest.mark.parametrize("test_params, expected_warn_msg", [
        # Vanilla test case
        ({
             "extractor_expt_date": datetime.datetime.strptime(
                 "2021-03-14 03:14:15",
                 "%Y-%m-%d %H:%M:%S"),
             "pkl_expt_date": datetime.datetime.strptime("2021-03-14 03:14:15",
                                                         "%Y-%m-%d %H:%M:%S"),
             "behavior_session_id": 1
         }, None),

        # pkl expt date stored in unix format
        ({
             "extractor_expt_date": datetime.datetime.strptime(
                 "2021-03-14 03:14:15",
                 "%Y-%m-%d %H:%M:%S"),
             "pkl_expt_date": 1615716855.0,
             "behavior_session_id": 2
         }, None),

        # Extractor and pkl dates differ significantly
        ({
             "extractor_expt_date": datetime.datetime.strptime(
                 "2021-03-14 03:14:15",
                 "%Y-%m-%d %H:%M:%S"),
             "pkl_expt_date": datetime.datetime.strptime("2021-03-14 20:14:15",
                                                         "%Y-%m-%d %H:%M:%S"),
             "behavior_session_id": 3
         },
         "The `date_of_acquisition` field in LIMS *"),

        # pkl file contains an unparseable datetime
        ({
             "extractor_expt_date": datetime.datetime.strptime(
                 "2021-03-14 03:14:15",
                 "%Y-%m-%d %H:%M:%S"),
             "pkl_expt_date": None,
             "behavior_session_id": 4
         },
         "Could not parse the acquisition datetime *"),
    ])
    def test_get_date_of_acquisition(self, tmp_path, test_params,
                                     expected_warn_msg):
        mock_session_id = test_params["behavior_session_id"]

        pkl_save_path = tmp_path / f"mock_pkl_{mock_session_id}.pkl"
        with open(pkl_save_path, 'wb') as handle:
            pickle.dump({"start_time": test_params['pkl_expt_date']}, handle)

        tz = pytz.timezone("America/Los_Angeles")
        extractor_expt_date = tz.localize(
            test_params['extractor_expt_date']).astimezone(pytz.utc)

        stimulus_file = StimulusFile(filepath=pkl_save_path)
        obt_date = DateOfAcquisition(
            date_of_acquisition=extractor_expt_date)

        if expected_warn_msg:
            with pytest.warns(Warning, match=expected_warn_msg):
                obt_date.validate(
                    stimulus_file=stimulus_file,
                    behavior_session_id=test_params['behavior_session_id'])

        assert obt_date.value == extractor_expt_date

    def test_indicator(self):
        """Test that indicator is parsed from full_genotype"""
        reporter_line = ReporterLine(
            reporter_line='Ai148(TIT2L-GC6f-ICL-tTA2)')
        assert reporter_line.parse_indicator() == 'GCaMP6f'

    @pytest.mark.parametrize("input_reporter_line, warning_msg, expected", (
            (None,
             'Could not parse indicator from reporter because there is no '
             'reporter', None),
            ('foo', 'Could not parse indicator from reporter because none'
                    'of the expected substrings were found in the reporter',
             None)
    )
                             )
    def test_indicator_edge_cases(self, input_reporter_line, warning_msg,
                                  expected):
        """Test indicator parsing edge cases"""
        with pytest.warns(UserWarning) as record:
            reporter_line = ReporterLine(reporter_line=input_reporter_line)
            indicator = reporter_line.parse_indicator(warn=True)
        assert indicator is expected
        assert str(record[0].message) == warning_msg


class TestStimulusFile:
    """Tests properties read from stimulus file"""
    def setup_class(cls):
        dir = Path(__file__).parent.parent.parent.resolve()
        test_data_dir = dir / 'test_data'
        sf_path = test_data_dir / 'stimulus_file.pkl'
        cls.stimulus_file = StimulusFile.from_json(
            dict_repr={'behavior_stimulus_file': str(sf_path)})

    def test_session_uuid(self):
        uuid = BehaviorSessionUUID.from_stimulus_file(
            stimulus_file=self.stimulus_file)
        expected = UUID('138531ab-fe59-4523-9154-07c8d97bbe03')
        assert expected == uuid.value

    def test_get_stimulus_frame_rate(self):
        rate = StimulusFrameRate.from_stimulus_file(
            stimulus_file=self.stimulus_file)
        assert 62.0 == rate.value


def test_date_of_acquisition_utc():
    """Tests that when read from json (in Pacific time), that
    date of acquisition is converted to utc"""
    expected = DateOfAcquisition(
        date_of_acquisition=datetime.datetime(2019, 9, 26, 16,
                                              tzinfo=pytz.UTC))
    actual = DateOfAcquisition.from_json(
        dict_repr={'date_of_acquisition': '2019-09-26 09:00:00'})
    assert expected == actual


class TestNWB(BehaviorMetaTestCase):
    def setup_method(self, method):
        self.nwbfile = pynwb.NWBFile(
            session_description='asession',
            identifier='afile',
            session_start_time=datetime.datetime.now()
        )

    @pytest.mark.parametrize('roundtrip', [True, False])
    def test_add_behavior_only_metadata(self, roundtrip,
                                        data_object_roundtrip_fixture):
        self.meta.to_nwb(nwbfile=self.nwbfile)

        if roundtrip:
            meta_obt = data_object_roundtrip_fixture(
                self.nwbfile, BehaviorMetadata
            )
        else:
            meta_obt = BehaviorMetadata.from_nwb(nwbfile=self.nwbfile)

        assert self.meta == meta_obt
