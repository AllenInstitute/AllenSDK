from . import PostgresQueryMixin
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def get_all_mesoscope_sessions():

    db = PostgresQueryMixin()
    query = ("select os.id as session_id, "
             "os.storage_directory as session_folder, "
             "sp.name as specimen, "
             "os.date_of_acquisition as date, "
             "os.workflow_state as session_wfl_state, " 
             "users.login as operator, "
             "p.code as project "
             "from ophys_sessions os "
             "join specimens sp on sp.id = os.specimen_id "
             "join projects p on p.id = os.project_id "
             "join users on users.id = os.operator_id "
             "join equipment rigs on rigs.id = os.equipment_id "
             "where rigs.name in ('MESO.1', 'MESO.2')"
             "order by date")
    return pd.read_sql(query, db.get_connection())
