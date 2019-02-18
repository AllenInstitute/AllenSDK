import psycopg2
import pandas as pd

from . import PostgresQueryMixin, one

class MtrainApi(PostgresQueryMixin):

    def __init__(self, dbname="mtrain", user="mtrainreader", host="prodmtrain1", password="mtrainro", port=5432):
        super(MtrainApi, self).__init__(dbname=dbname, user=user, host=host, password=password, port=port)

    def get_subjects(self):
        query = 'SELECT "LabTracks_ID" FROM subjects'
        return self.fetchall(query)


    def get_behavior_training_df(self, LabTracks_ID):
        connection = self.get_connection()
        dataframe = pd.read_sql(
            '''SELECT stages.name as stage_name, regimens.name as regimen_name, bs.date, bs.id as behavior_session_id
               FROM behavior_sessions bs
               LEFT JOIN states ON states.id = bs.state_id
               LEFT JOIN regimens ON regimens.id = states.regimen_id
               LEFT JOIN stages ON stages.id = states.stage_id
               WHERE "LabTracks_ID"={}
            '''.format(LabTracks_ID), connection)
        return dataframe.sort_values(by='date')
