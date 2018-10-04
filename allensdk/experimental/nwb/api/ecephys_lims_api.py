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

        df = self.query_fn(query)
        return df


    def get_probe_table(self, session_ids=None):
        '''
        '''

        query = clean_multiline_query('''
            select * from ecephys_probes ep
        ''')

        if session_ids is not None:
            query = '{} where ep.ecephys_session_id in {}'.format(query, produce_in_clause_target(session_ids))

        df = self.query_fn(query)
        return df


    def get_unit_table(self, session_id, probe_ids=None):
        '''
        '''

        probe_ids = [
            pid for pid in self.get_probe_table(session_ids=[session_id])['id'].values 
            if probe_ids is None or pid in probe_ids
        ] #  O(n**2), but max about 6 - might change if we can fit more probes in a brain

        files = self.get_well_known_file_table(
            attachable_types=['EcephysProbe'],
            attachable_ids=probe_ids,
            file_type_names=['EcephysSortedClusterGroup', 'EcephysSortedMetrics']
        )
        channel_table = self.get_channel_table(session_id, probe_ids=probe_ids)
        channel_table = channel_table.loc[:, ('probe_id', 'local_index', 'id')]

        unit_table = []
        current_id = 0
        for ii, (index, probe_files) in enumerate(files.groupby(['attachable_id'])):
            probe_id = probe_files['attachable_id'].values[0]

            cg_path = probe_files[probe_files['file_type_name'] == 'EcephysSortedClusterGroup']['path'].values[0]
            try:
                metrics_path = probe_files[probe_files['file_type_name'] == 'EcephysSortedMetrics']['path'].values[0]
            except IndexError: # TODO: for some reason some probes have metrics files in place, but no records
                metrics_path = os.path.join(os.path.dirname(cg_path), 'metrics.csv')

            cluster_groups = pd.read_csv(cg_path, delim_whitespace=True, index_col=False)
            metrics = pd.read_csv(metrics_path, index_col=0)

            probe_channels = channel_table[channel_table['probe_id'] == probe_id]

            probe_unit_table = pd.DataFrame({
                'id': metrics['cluster_ids'].values + current_id,
                'local_peak_channel_index': metrics['peak_chan'].values,
                'quality': metrics['unit_quality'],
                'snr': metrics['snr'].values,
                'firing_rate': metrics['firing_rate'].values,
                'isi_violations': metrics['isi_viol'].values,
            })
            probe_unit_table = probe_unit_table.merge(probe_channels, left_on='local_peak_channel_index', right_on='local_index', suffixes=['', '_channel'])
            probe_unit_table = probe_unit_table.drop(columns=['local_peak_channel_index', 'local_index'])
            probe_unit_table = probe_unit_table.rename(columns=lambda colname: 'peak_channel_id' if colname == 'id_channel' else colname)

            unit_table.append(probe_unit_table)
            current_id = np.amax(probe_unit_table['id']) + 1

        unit_table = pd.concat(unit_table)
        unit_table = unit_table.sort_values(by='id')
        unit_table = unit_table.reset_index(drop=True)
        return unit_table
        


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
            query = '{} and ep.id in {}'.format(query, produce_in_clause_target(probe_ids))
        response = self.query_fn(query).to_dict('record')

        probe_dfs = []
        last_num_channels = 0
        for ii, probe in enumerate(response):
            max_vertical_pos = np.amax(probe['probe_info']['vertical_pos'])
            num_channels = len(probe['probe_info']['channel'])

            probe_df = pd.DataFrame({
                'id': np.array(probe['probe_info']['channel']) + ii * last_num_channels, #  TODO: we don't really have ids for these ...
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
        df.reset_index(drop=True)
        return df


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


    def get_probe_and_session_well_known_files(self):
        '''Queries for a global table of all well known files and their paths, for all experiments and probes

        Notes
        -----
        only valid for LIMS

        '''

        session_query = clean_multiline_query('''
            select wkft.name as file_type, wkf.storage_directory, wkf.filename, wkf.attachable_id as session_id
            from well_known_files wkf
            join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
            where wkf.attachable_type = \'EcephysSession\'
        ''')
        session_response = self.query_fn(session_query)
        session_response['probe_id'] = None

        probe_query = clean_multiline_query('''
            select wkft.name as file_type, wkf.storage_directory, wkf.filename, wkf.attachable_id as probe_id, ep.ecephys_session_id as session_id
            from well_known_files wkf
            join ecephys_probes ep on ep.id = wkf.attachable_id
            join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
            where wkf.attachable_type = \'EcephysProbe\'
        ''')
        probe_response = self.query_fn(probe_query)
        
        output = pd.concat([session_response, probe_response], sort=False)
        output['path'] = output.apply(lambda row: os.path.join(row['storage_directory'], row['filename']), axis=1)
        output = output.drop(columns=['storage_directory', 'filename'])
        return output.sort_values(by='session_id')