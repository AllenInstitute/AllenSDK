import datetime
import uuid

import pytz

from allensdk.brain_observatory.behavior.data_objects import BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_metadata import \
    BehaviorMetadata
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.behavior_session_uuid import \
    BehaviorSessionUUID
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.date_of_acquisition import \
    DateOfAcquisition
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.equipment_name import \
    EquipmentName
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.session_type import \
    SessionType
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .behavior_metadata.stimulus_frame_rate import \
    StimulusFrameRate
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.age import \
    Age
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.driver_line import \
    DriverLine
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.full_genotype import \
    FullGenotype
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.mouse_id import \
    MouseId
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.reporter_line import \
    ReporterLine
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.sex import \
    Sex
from allensdk.brain_observatory.behavior.data_objects.metadata\
    .subject_metadata.subject_metadata import \
    SubjectMetadata


class TestBehaviorMetadata:
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
            equipment_name=EquipmentName(equipment_name='my_device'),
            stimulus_frame_rate=StimulusFrameRate(stimulus_frame_rate=60.0),
            session_type=SessionType(session_type='Unknown'),
            date_of_acquisition=DateOfAcquisition(
                date_of_acquisition=pytz.utc.localize(datetime.datetime.now())
            ),
            behavior_session_uuid=BehaviorSessionUUID(
                behavior_session_uuid=uuid.uuid4())
        )
        return behavior_meta