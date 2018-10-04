import functools
import os
import warnings

import pandas as pd
import numpy as np

from .ecephys_api import EcephysApi
from .lims_api import LimsApi, clean_multiline_query, produce_in_clause_target


class EcephysLimsApi(LimsApi, EcephysApi):

    def get_session_table(self, session_ids=None):
        '''
        '''

        query = clean_multiline_query('''
            select es.id, es.name, es.workflow_state, es.specimen_id, es.project_id, es.observatory_stimulus_config_id, es.date_of_acquisition 
            from ecephys_sessions es
        ''')

        if session_ids is not None:
            query = '{} where es.id in {}'.format(query, produce_in_clause_target(session_ids))

        return self.query_fn(query)


    def get_probe_table(self, session_ids=None):
        '''
        '''

        query = clean_multiline_query('''
            select * from ecephys_probes ep
        ''')

        if session_ids is not None:
            query = '{} where ep.ecephys_session_id in {}'.format(query, produce_in_clause_target(session_ids))

        df = self.query_fn(query)
        df = df.set_index('id')
        return df


    def get_stimulus_table(self, session_id):
        path = self._get_well_known_file_path(
            attachable_id=session_id, well_known_file_type_name='EcephysStimulusTable', attachable_type='EcephysSession'
        )
        return pd.read_csv(path)


    def get_channel_table(self, session_id, probe_ids=None):
        '''

        Notes
        -----
        This needs to be modeled properly in LIMS.

        '''

        query = clean_multiline_query('''
            select * from ecephys_probes ep
            where ep.ecephys_session_id = {}
        '''.format(session_id))

        if probe_ids is not None:
            query = '{} where ep.id in {}'.format(query, produce_in_clause_target(probe_ids))

        response = self.query_fn(query).to_dict('record')
        probe_dfs = []
        last_num_channels = 0
        for ii, probe in enumerate(response):
            max_vertical_pos = np.amax(probe['probe_info']['vertical_pos'])
            num_channels = len(probe['probe_info']['channel'])

            probe_df = pd.DataFrame({
                'id': np.array(probe['probe_info']['channel']) + ii * last_num_channels, # we don't really have ids for these ...
                'local_index': probe['probe_info']['channel'],
                'probe_id': np.zeros(num_channels, dtype=int) + probe['id'],
                'mask': probe['probe_info']['mask'],
                'vertical_pos': np.array(probe['probe_info']['vertical_pos']) - max_vertical_pos,
                'horizontal_pos': probe['probe_info']['horizontal_pos']
            })

            probe_dfs.append(probe_df)
            last_num_channels = num_channels

        df = pd.concat(probe_dfs)
        df = df.sort_values(by='id')
        df.reset_index()
        return df.set_index('id')


    def get_lims_labtracks_map(self):
        '''

        Notes
        -----
        only valid for LIMS

        '''

        query = clean_multiline_query('''
            select es.id, sp.external_specimen_name from ecephys_sessions es
            join specimens sp on sp.id = es.specimen_id
        ''')
        response = self.query_fn(query)
        response['external_specimen_name'] = response['external_specimen_name'].astype(int)
        return response 


