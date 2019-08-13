from . import PostgresQueryMixin
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def get_all_mesoscope_data():

    db = PostgresQueryMixin()
    query = ("select os.id as session_id, oe.id as experiment_id, "
             "os.storage_directory as session_folder, oe.storage_directory as experiment_folder, "
             "sp.name as specimen, "
             "os.date_of_acquisition as date, "
             "oe.workflow_state as exp_wfl_state, "
             "os.workflow_state as session_wfl_state, " 
             "users.login as operator, "
             "p.code as project "
             "from ophys_experiments oe "
             "join ophys_sessions os on os.id = oe.ophys_session_id "
             "join specimens sp on sp.id = os.specimen_id "
             "join projects p on p.id = os.project_id "
             "join users on users.id = os.operator_id "
             "join equipment rigs on rigs.id = os.equipment_id "
             "where rigs.name in ('MESO.1', 'MESO.2')"
             "order by date")
    return pd.read_sql(query, db.get_connection())
