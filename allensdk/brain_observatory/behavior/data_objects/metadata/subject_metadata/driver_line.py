from typing import List

from pynwb import NWBFile

from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base \
    .readable_interfaces import \
    JsonReadableInterface, LimsReadableInterface, NwbReadableInterface
from allensdk.internal.api import PostgresQueryMixin, \
    OneOrMoreResultExpectedError


class DriverLine(DataObject, LimsReadableInterface, JsonReadableInterface,
                 NwbReadableInterface):
    """the genotype name(s) of the driver line(s)"""
    def __init__(self, driver_line: List[str]):
        super().__init__(name="driver_line", value=driver_line)

    @classmethod
    def from_json(cls, dict_repr: dict) -> "DriverLine":
        return cls(driver_line=dict_repr['driver_line'])

    @classmethod
    def from_lims(cls, behavior_session_id: int,
                  lims_db: PostgresQueryMixin) -> "DriverLine":
        query = f"""
            SELECT g.name AS driver_line
            FROM behavior_sessions bs
            JOIN donors d ON bs.donor_id=d.id
            JOIN donors_genotypes dg ON dg.donor_id=d.id
            JOIN genotypes g ON g.id=dg.genotype_id
            JOIN genotype_types gt
                ON gt.id=g.genotype_type_id AND gt.name = 'driver'
            WHERE bs.id={behavior_session_id};
        """
        result = lims_db.fetchall(query)
        if result is None or len(result) < 1:
            raise OneOrMoreResultExpectedError(
                f"Expected one or more, but received: '{result}' "
                f"from query:\n'{query}'")
        driver_line = sorted(result)
        return cls(driver_line=driver_line)

    @classmethod
    def from_nwb(cls, nwbfile: NWBFile) -> "DriverLine":
        driver_line = sorted(list(nwbfile.subject.driver_line))
        return cls(driver_line=driver_line)
