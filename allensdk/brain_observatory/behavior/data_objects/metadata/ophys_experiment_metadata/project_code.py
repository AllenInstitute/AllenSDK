from allensdk.brain_observatory.behavior.data_objects import DataObject
from allensdk.brain_observatory.behavior.data_objects.base\
    .readable_interfaces import \
    LimsReadableInterface
from allensdk.internal.api import PostgresQueryMixin


class ProjectCode(DataObject, LimsReadableInterface):
    def __init__(self, project_code: str):
        super().__init__(name='project_code', value=project_code)

    @classmethod
    def from_lims(cls, ophys_experiment_id: int,
                  lims_db: PostgresQueryMixin) -> "ProjectCode":
        query = f"""
            SELECT projects.code AS project_code
            FROM ophys_sessions
            JOIN projects ON projects.id = ophys_sessions.project_id
            WHERE ophys_sessions.id = (
                SELECT oe.ophys_session_id
                FROM ophys_experiments oe
                WHERE oe.id = {ophys_experiment_id}
            )
        """
        project_code = lims_db.fetchone(query, strict=True)
        return cls(project_code=project_code)
