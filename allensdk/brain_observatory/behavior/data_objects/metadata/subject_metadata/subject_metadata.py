from typing import Optional, List

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject, \
    BehaviorSessionId
from allensdk.brain_observatory.behavior.data_objects._base.readable_interfaces\
    .lims_readable_interface import \
    LimsReadableInterface
from allensdk.brain_observatory.behavior.data_objects._base.writable_interfaces\
    .nwb_writable_interface import \
    NwbWritableInterface
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
from allensdk.brain_observatory.behavior.schemas import SubjectMetadataSchema
from allensdk.brain_observatory.nwb import load_pynwb_extension
from allensdk.internal.api import PostgresQueryMixin


class SubjectMetadata(DataObject, LimsReadableInterface,
                      NwbWritableInterface):
    """Subject metadata"""
    def __init__(self,
                 sex: Sex,
                 age: Age,
                 reporter_line: ReporterLine,
                 full_genotype: FullGenotype,
                 driver_line: DriverLine,
                 mouse_id: MouseId):
        super().__init__(name='subject_metadata', value=self)
        self._sex = sex
        self._age = age
        self._reporter_line = reporter_line
        self._full_genotype = full_genotype
        self._driver_line = driver_line
        self._mouse_id = mouse_id

    @classmethod
    def from_lims(cls,
                  behavior_session_id: BehaviorSessionId,
                  lims_db: PostgresQueryMixin) -> "SubjectMetadata":
        sex = Sex.from_lims(behavior_session_id=behavior_session_id.value,
                            lims_db=lims_db)
        age = Age.from_lims(behavior_session_id=behavior_session_id.value,
                            lims_db=lims_db)
        reporter_line = ReporterLine.from_lims(
            behavior_session_id=behavior_session_id.value, lims_db=lims_db)
        full_genotype = FullGenotype.from_lims(
            behavior_session_id=behavior_session_id.value, lims_db=lims_db)
        driver_line = DriverLine.from_lims(
            behavior_session_id=behavior_session_id.value, lims_db=lims_db)
        mouse_id = MouseId.from_lims(
            behavior_session_id=behavior_session_id.value,
                                     lims_db=lims_db)
        return cls(
            sex=sex,
            age=age,
            full_genotype=full_genotype,
            driver_line=driver_line,
            mouse_id=mouse_id,
            reporter_line=reporter_line
        )

    def to_json(self) -> dict:
        pass

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "SubjectMetadata":
        mouse_id = MouseId.from_nwb(nwbfile=nwbfile)
        sex = Sex.from_nwb(nwbfile=nwbfile)
        age = Age.from_nwb(nwbfile=nwbfile)
        reporter_line = ReporterLine.from_nwb(nwbfile=nwbfile)
        driver_line = DriverLine.from_nwb(nwbfile=nwbfile)
        genotype = FullGenotype.from_nwb(nwbfile=nwbfile)

        return cls(
            mouse_id=mouse_id,
            sex=sex,
            age=age,
            reporter_line=reporter_line,
            driver_line=driver_line,
            full_genotype=genotype
        )

    @classmethod
    def to_nwb(cls, nwbfile: NWBFile) -> NWBFile:
        BehaviorSubject = load_pynwb_extension(SubjectMetadataSchema,
                                               'ndx-aibs-behavior-ophys')
        nwb_subject = BehaviorSubject(
            description="A visual behavior subject with a LabTracks ID",
            age=Age.to_iso8601(age=cls.age_in_days),
            driver_line=cls.driver_line,
            genotype=cls.full_genotype,
            subject_id=str(cls.mouse_id),
            reporter_line=cls.reporter_line,
            sex=cls.sex,
            species='Mus musculus')
        nwbfile.subject = nwb_subject
        return nwbfile

    @property
    def sex(self) -> str:
        return self._sex.value

    @property
    def age_in_days(self) -> Optional[int]:
        return self._age.value

    @property
    def reporter_line(self) -> Optional[str]:
        return self._reporter_line.value

    @property
    def indicator(self) -> Optional[str]:
        return self._reporter_line.indicator

    @property
    def full_genotype(self) -> str:
        return self._full_genotype.value

    @property
    def cre_line(self) -> Optional[str]:
        return self._full_genotype.parse_cre_line(warn=True)

    @property
    def driver_line(self) -> List[str]:
        return self._driver_line.value

    @property
    def mouse_id(self) -> int:
        return self._mouse_id.value