import os
import shutil
import warnings

import pandas as pd

from allensdk.api.cache import Cache, cacheable
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

from .ecephys_project_api import EcephysProjectApi
from .lims_api_mixin import LimsApiMixin


csv_io = {
    'reader': lambda path: pd.read_csv(path, index_col='id'),
    'writer': lambda path, df: df.to_csv(path)
}


class EcephysProjectLimsApi(EcephysProjectApi, LimsApiMixin):

    def __init__(self, **kwargs):
        super(EcephysProjectApi, self).__init__(**kwargs)

    @cacheable(
        strategy='lazy', 
        pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'),
        reader=EcephysSession.from_nwb_path
    )
    def get_session_data(self, path, session_id):
        nwb_paths = self._get_session_nwb_paths(session_id)
        main_nwb_path = nwb_paths.loc[nwb_paths['name'] == 'EcephysNwb']['path'].values

        if len(main_nwb_path) == 1 and not isinstance(main_nwb_path, str):
            main_nwb_path = main_nwb_path[0]
        else:
            raise ValueError(f'did not find a unique nwb path for session {session_id}')

        fsize = os.path.getsize(main_nwb_path) / 1024 ** 2
        warnings.warn(f'copying a {fsize:.6}mb file from {main_nwb_path} to {path}')
        shutil.copyfile(main_nwb_path, path)
        return path

    @cacheable(
        strategy='lazy',
        pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'),
        **csv_io
    )
    def get_units(self, path, **kwargs):
        return self._get_units(**kwargs)

    @cacheable(
        strategy='lazy',
        pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'),
        **csv_io
    )
    def get_channels(self, path, **kwargs):
        return self._get_channels(**kwargs)

    @cacheable(
        strategy='lazy',
        pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'),
        **csv_io
    )
    def get_probes(self, path, **kwargs):
        return self._get_probes(**kwargs)

    @cacheable(
        strategy='lazy', 
        pathfinder=Cache.pathfinder(file_name_position=1, path_keyword='path'),
        **csv_io
    )
    def get_sessions(self, path, **kwargs):
        return self._get_sessions(**kwargs)

    def _get_units(self, unit_ids=None, channel_ids=None, probe_ids=None, session_ids=None):

        filters = []
        filters.append(containment_filter_clause(channel_ids, 'eu.id'))
        filters.append(containment_filter_clause(channel_ids, 'ec.id'))
        filters.append(containment_filter_clause(probe_ids, 'ep.id'))
        filters.append(containment_filter_clause(session_ids, 'es.id'))

        query = f'''
            select eu.* from ecephys_units eu
            join ecephys_channels ec on ec.id = eu.ecephys_channel_id
            join ecephys_probes ep on ep.id = ec.ecephys_probe_id
            join ecephys_sessions es on es.id = ep.ecephys_session_id 
            {and_filters(filters)}
        '''

        results = self.select(query)
        results.set_index('id', inplace=True)

        return results

    def _get_channels(self, channel_ids=None, probe_ids=None, session_ids=None):

        filters = []
        filters.append(containment_filter_clause(channel_ids, 'ec.id'))
        filters.append(containment_filter_clause(probe_ids, 'ep.id'))
        filters.append(containment_filter_clause(session_ids, 'es.id'))

        query = f'''
            select ec.* from ecephys_channels ec
            join ecephys_probes ep on ep.id = ec.ecephys_probe_id
            join ecephys_sessions es on es.id = ep.ecephys_session_id 
            {and_filters(filters)}
        '''

        results = self.select(query)
        results.set_index('id', inplace=True)

        return results

    def _get_probes(self, probe_ids=None, session_ids=None, workflow_states=('uploaded',)):

        filters = []
        filters.append(containment_filter_clause(probe_ids, 'ep.id'))
        filters.append(containment_filter_clause(session_ids, 'es.id'))

        query = f'''
            select ep.* from ecephys_probes ep 
            join ecephys_sessions es on es.id = ep.ecephys_session_id 
            {and_filters(filters)}
        '''

        results = self.select(query)
        results.set_index('id', inplace=True)
        results.drop(columns='probe_info', inplace=True)

        return results

    def _get_sessions(self, 
        session_ids=None, 
        workflow_states=('uploaded',),
        published=None,
        habituation=False,
        project_names=('BrainTV Neuropixels Visual Behavior', 'BrainTV Neuropixels Visual Coding')
    ):

        filters = []
        filters.append(containment_filter_clause(session_ids, 'es.id'))
        filters.append(containment_filter_clause(project_names, 'es.workflow_state', True))
        filters.append(containment_filter_clause(project_names, 'pr.id', True))

        if published is not None:
            filters.append(f'es.published_at is {"not" if published else ""} null')
        if habituation is False:
            filters.append('habituation = false')
        elif habituation is True:
            filters.append('habituation = true')

        query = f'''
            select es.* from ecephys_sessions es 
            join project_ids pr on pr.id = es.project_id {and_filters(filters)}
        '''
        response = self.select(query)
        response.set_index('id', inplace=True)

        return response

    def _get_session_nwb_paths(self, session_id):

        probe_response = self._get_probe_well_known_files([session_id], wkf_types=['EcephysLfpNwb'])
        session_response = self._get_session_well_known_files([session_id], ['EcephysNwb'])

        session_response['ecephys_probe_id'] = None
        response = (pd.concat([session_response, probe_response], sort=False)
            .reset_index()
            .drop(columns='index'))

        return response
        
    def _get_probe_well_known_files(self, session_ids=None, probe_ids=None, wkf_types=None):
        
        filters = []
        filters.append(containment_filter_clause(session_ids, 'es.id'))
        filters.append(containment_filter_clause(probe_ids, 'ep.id'))
        filters.append(containment_filter_clause(wkf_types, 'wkft.name'), True)

        # TODO: why does the probe analysis runs table not have a "current" field?
        query = f'''
            select wkf.storage_directory, wkf.filename, wkft.name, 
            es.id as ecephys_session_id, ep.id as ecephys_probe_id from ecephys_sessions es
            join ecephys_probes ep on ep.ecephys_session_id = es.id
            join ecephys_analysis_run_probes earp on (
                earp.ecephys_probe_id = ep.id
            )
            join well_known_files wkf on wkf.attachable_id = earp.id
            join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
            {and_filters(filters)}
        '''

        response = self.select(query)
        return clean_wkf_response(response)

    def _get_session_well_known_files(self, session_ids=None, wkf_types=None):

        filters = []
        filters.append(containment_filter_clause(session_ids), 'es.id')
        filters.append(containment_filter_clause(wkf_types, 'wkft.name'), True)

        query = f''' 
            select wkf.storage_directory, wkf.filename, es.id as ecephys_session_id, wkft.name from ecephys_sessions es
            join ecephys_analysis_runs ear on (
                ear.ecephys_session_id = es.id
                and ear.current
            )
            join well_known_files wkf on wkf.attachable_id = ear.id
            join well_known_file_types wkft on wkft.id = wkf.well_known_file_type_id
            {and_filters(filters)}
        '''

        response = self.select(query)
        return clean_wkf_response(response)


def containment_filter_clause(pass_values, field, quote=False):
    if pass_values is None or len(pass_values) == 0:
        return ''

    if quote:
        pass_values = [f"'{p}'" for p in pass_values]
    else:
        pass_values = [f"{p}" for p in pass_values]
    
    return f'{field} in ({",".join(pass_values)})'


def and_filters(filters):
    filters = [ff for ff in filters if ff]
    return f'where {" and ".join(filters)}' if filters else ''


def clean_wkf_response(response):
    if response.shape[0] == 0:
        return response
    response['path'] = response.apply(lambda row: os.path.join(row['storage_directory'], row['filename']), axis=1)
    response.drop(columns=['storage_directory', 'filename'], inplace=True)
    return response
