import re
import warnings
from typing import Optional

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.internal.api import PostgresQueryMixin


class Age(DataObject):
    """Age of animal (in days)"""
    def __init__(self, age: int):
        super().__init__(name="age_in_days", value=age)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "Age":
        age = dict_repr["age"]
        age = cls._age_code_to_days(age=age)
        return cls(age=age)

    def to_json(self) -> dict:
        return {"sex": self.value}

    @classmethod
    def from_lims(cls, behavior_session_id: int,
                  lims_db: PostgresQueryMixin) -> "Age":
        query = f"""
            SELECT a.name AS age
            FROM behavior_sessions bs
            JOIN donors d ON d.id = bs.donor_id
            JOIN ages a ON a.id = d.age_id
            WHERE bs.id = {behavior_session_id};
        """
        age = lims_db.fetchone(query, strict=True)
        age = cls._age_code_to_days(age=age)
        return cls(age=age)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "EquipmentName":
        pass

    def to_nwb(self, nwbfile: NWBFile) -> NWBFile:
        pass

    @staticmethod
    def _age_code_to_days(age: str, warn=False) -> Optional[int]:
        """Converts the age code into a numeric days representation

        Parameters
        ----------
        age
            age code, ie P123
        warn
            Whether to output warning if parsing fails
        """
        if not age.startswith('P'):
            if warn:
                warnings.warn('Could not parse numeric age from age code '
                              '(age code does not start with "P")')
            return None

        match = re.search(r'\d+', age)

        if match is None:
            if warn:
                warnings.warn('Could not parse numeric age from age code '
                              '(no numeric values found in age code)')
            return None

        start, end = match.span()
        return int(age[start:end])
