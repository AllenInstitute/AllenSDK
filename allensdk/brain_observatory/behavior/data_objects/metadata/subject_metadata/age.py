import re
import warnings
from typing import Optional

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.internal.api import PostgresQueryMixin


class Age(DataObject, JsonReadableInterface, LimsReadableInterface,
          NwbReadableInterface):
    """Age of animal (in days)"""
    def __init__(self, age: int):
        super().__init__(name="age_in_days", value=age)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "Age":
        age = dict_repr["age"]
        age = cls._age_code_to_days(age=age)
        return cls(age=age)

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
    def from_nwb(cls, nwbfile: NWBFile) -> "Age":
        age = cls._age_code_to_days(age=nwbfile.subject.age)
        return cls(age=age)

    @staticmethod
    def to_iso8601(age: int):
        if age is None:
            return 'null'
        return f'P{age}D'

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
