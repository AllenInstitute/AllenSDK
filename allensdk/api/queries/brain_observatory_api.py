# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import logging

import pandas as pd
from six import string_types

from allensdk.config.manifest import Manifest
import allensdk.brain_observatory.stimulus_info as stimulus_info

from .rma_template import RmaTemplate
from ..cache import cacheable, Cache
from .rma_pager import pageable


class BrainObservatoryApi(RmaTemplate):
    _log = logging.getLogger('allensdk.api.queries.brain_observatory_api')

    NWB_FILE_TYPE = 'NWBOphys'
    CELL_MAPPING_ID = 590985414

    rma_templates = \
        {"brain_observatory_queries": [
            {'name': 'list_isi_experiments',
             'description': 'see name',
             'model': 'IsiExperiment',
             'num_rows': 'all',
             'count': False,
             'criteria_params': []
             },
            {'name': 'isi_experiment_by_ids',
             'description': 'see name',
             'model': 'IsiExperiment',
             'criteria': '[id$in{{ isi_experiment_ids }}]',
             'include': 'experiment_container(ophys_experiments,targeted_structure)',
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['isi_experiment_ids']
             },
            {'name': 'ophys_experiment_by_ids',
             'description': 'see name',
             'model': 'OphysExperiment',
             'criteria': '{% if ophys_experiment_ids is defined %}[id$in{{ ophys_experiment_ids }}]{%endif%}',
             'include': 'experiment_container,well_known_files(well_known_file_type),targeted_structure,specimen(donor(age,transgenic_lines))',
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['ophys_experiment_ids']
             },
            {'name': 'ophys_experiment_data',
             'description': 'see name',
             'model': 'WellKnownFile',
             'criteria': '[attachable_id$eq{{ ophys_experiment_id }}],well_known_file_type[name$eq%s]' % NWB_FILE_TYPE,
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['ophys_experiment_id']
             },
            {'name': 'column_definitions',
             'description': 'see name',
             'model': 'ApiColumnDefinition',
             'criteria': '[api_class_name$eq{{ api_class_name }}]',
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['api_class_name']
             },
            {'name': 'column_definition_class_names',
             'description': 'see name',
             'model': 'ApiColumnDefinition',
             'only': ['api_class_name'],
             'num_rows': 'all',
             'count': False,
             },
            {'name': 'stimulus_mapping',
             'description': 'see name',
             'model': 'ApiCamStimulusMapping',
             'criteria': '{% if stimulus_mapping_ids is defined %}[id$in{{ stimulus_mapping_ids }}]{%endif%}',
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['stimulus_mapping_ids']
             },
            {'name': 'experiment_container',
             'description': 'see name',
             'model': 'ExperimentContainer',
             'criteria': '{% if experiment_container_ids is defined %}[id$in{{ experiment_container_ids }}]{%endif%}',
             'include': 'ophys_experiments,isi_experiment,specimen(donor(conditions,age,transgenic_lines)),targeted_structure',
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['experiment_container_ids']
             },
            {'name': 'experiment_container_metric',
             'description': 'see name',
             'model': 'ApiCamExperimentContainerMetric',
             'criteria': '{% if experiment_container_metric_ids is defined %}[id$in{{ experiment_container_metric_ids }}]{%endif%}',
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['experiment_container_metric_ids']
             },
            {'name': 'cell_metric',
             'description': 'see name',
             'model': 'ApiCamCellMetric',
             'criteria': '{% if cell_specimen_ids is defined %}[cell_specimen_id$in{{ cell_specimen_ids }}]{%endif%}',
             'criteria_params': ['cell_specimen_ids']
             },
            {'name': 'cell_specimen_id_mapping_table',
             'description': 'see name',
             'model': 'WellKnownFile',
             'criteria': '[id$eq{{ mapping_table_id }}],well_known_file_type[name$eqOphysCellSpecimenIdMapping]',
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['mapping_table_id']}
        ]}

    _QUERY_TEMPLATES = {
        "=": '({0} == {1})',
        "<": '({0} < {1})',
        ">": '({0} > {1})',
        "<=": '({0} <= {1})',
        ">=": '({0} >= {1})',
        "between": '({0} >= {1}) and ({0} <= {2})',
        "in": '({0} == {1})',
        "is": '({0} == {1})'
    }

    def __init__(self, base_uri=None, datacube_uri=None):
        super(BrainObservatoryApi, self).__init__(base_uri,
                                                  query_manifest=BrainObservatoryApi.rma_templates)

        self.datacube_uri = datacube_uri

    @cacheable()
    def get_ophys_experiments(self, ophys_experiment_ids=None):
        ''' Get OPhys Experiments by id

        Parameters
        ----------
        ophys_experiment_ids : integer or list of integers, optional
            only select specific experiments.

        Returns
        -------
        dict : ophys experiment metadata
        '''
        data = self.template_query('brain_observatory_queries',
                                   'ophys_experiment_by_ids',
                                   ophys_experiment_ids=ophys_experiment_ids)

        return data

    def get_isi_experiments(self, isi_experiment_ids=None):
        ''' Get ISI Experiments by id

        Parameters
        ----------
        isi_experiment_ids : integer or list of integers, optional
            only select specific experiments.

        Returns
        -------
        dict : isi experiment metadata
        '''
        data = self.template_query('brain_observatory_queries',
                                   'isi_experiment_by_ids',
                                   isi_experiment_ids=isi_experiment_ids)

        return data

    def list_isi_experiments(self, isi_ids=None):
        '''List ISI experiments available through the Allen Institute API

        Parameters
        ----------
        neuronal_model_ids : integer or list of integers, optional
            only select specific isi experiments.

        Returns
        -------
        dict : neuronal model metadata
        '''
        data = self.template_query('brain_observatory_queries',
                                   'list_isi_experiments')

        return data

    def list_column_definition_class_names(self):
        ''' Get column definitions

        Parameters
        ----------

        Returns
        -------
        list : api class name strings
        '''
        data = self.template_query('brain_observatory_queries',
                                   'column_definition_class_names')

        names = list(set([n['api_class_name'] for n in data]))

        return names

    def get_column_definitions(self, api_class_name=None):
        ''' Get column definitions

        Parameters
        ----------
        api_class_names : string or list of strings, optional
            only select specific column definition records.

        Returns
        -------
        dict : column definition metadata
        '''
        data = self.template_query('brain_observatory_queries',
                                   'column_definitions',
                                   api_class_name=api_class_name)

        return data

    @cacheable()
    def get_stimulus_mappings(self, stimulus_mapping_ids=None):
        ''' Get stimulus mappings by id

        Parameters
        ----------
        stimulus_mapping_ids : integer or list of integers, optional
            only select specific stimulus mapping records.

        Returns
        -------
        dict : stimulus mapping metadata
        '''
        data = self.template_query('brain_observatory_queries',
                                   'stimulus_mapping',
                                   stimulus_mapping_ids=stimulus_mapping_ids)

        return data

    @cacheable()
    @pageable(num_rows=2000, total_rows='all')
    def get_cell_metrics(self, cell_specimen_ids=None, *args, **kwargs):
        ''' Get cell metrics by id

        Parameters
        ----------
        cell_metrics_ids : integer or list of integers, optional
            only select specific cell metric records.

        Returns
        -------
        dict : cell metric metadata
        '''
        data = self.template_query('brain_observatory_queries',
                                   'cell_metric',
                                   cell_specimen_ids=cell_specimen_ids,
                                   *args,
                                   **kwargs)

        return data

    @cacheable()
    def get_experiment_containers(self, experiment_container_ids=None):
        ''' Get experiment container by id

        Parameters
        ----------
        experiment_container_ids : integer or list of integers, optional
            only select specific experiment containers.

        Returns
        -------
        dict : experiment container metadata
        '''
        data = self.template_query('brain_observatory_queries',
                                   'experiment_container',
                                   experiment_container_ids=experiment_container_ids)

        return data

    def get_experiment_container_metrics(self, experiment_container_metric_ids=None):
        ''' Get experiment container metrics by id

        Parameters
        ----------
        isi_experiment_ids : integer or list of integers, optional
            only select specific experiments.

        Returns
        -------
        dict : isi experiment metadata
        '''
        data = self.template_query('brain_observatory_queries',
                                   'experiment_container_metric',
                                   experiment_container_metric_ids=experiment_container_metric_ids)

        return data

    @cacheable(strategy='create',
               pathfinder=Cache.pathfinder(file_name_position=2,
                                           path_keyword='file_name'))
    def save_ophys_experiment_data(self, ophys_experiment_id, file_name):
        data = self.template_query('brain_observatory_queries',
                                   'ophys_experiment_data',
                                   ophys_experiment_id=ophys_experiment_id)

        try:
            file_url = data[0]['download_link']
        except Exception as _:
            raise Exception("ophys experiment %d has no data file" %
                            ophys_experiment_id)

        self._log.warning(
            "Downloading ophys_experiment %d NWB. This can take some time." % ophys_experiment_id)

        self.retrieve_file_over_http(self.api_url + file_url, file_name)

    def filter_experiment_containers(self, containers,
                                     ids=None,
                                     targeted_structures=None,
                                     imaging_depths=None,
                                     transgenic_lines=None,
                                     include_failed=False):

        if not include_failed:
            containers = [c for c in containers if not c.get('failed', False)]

        if ids is not None:
            containers = [c for c in containers if c['id'] in ids]

        if targeted_structures is not None:
            containers = [c for c in containers if c[
                'targeted_structure']['acronym'] in targeted_structures]

        if imaging_depths is not None:
            containers = [c for c in containers if c[
                'imaging_depth'] in imaging_depths]

        if transgenic_lines is not None:
            tls = [ tl.lower() for tl in transgenic_lines ]
            containers = [c for c in containers for tl in c['specimen'][
                'donor']['transgenic_lines'] if tl['name'].lower() in tls]

        return containers

    def filter_ophys_experiments(self, experiments,
                                 ids=None,
                                 experiment_container_ids=None,
                                 targeted_structures=None,
                                 imaging_depths=None,
                                 transgenic_lines=None,
                                 stimuli=None,
                                 session_types=None,
                                 include_failed=False,
                                 require_eye_tracking=False):

        # re-using the code from above
        experiments = self.filter_experiment_containers(experiments,
                                                        ids=ids,
                                                        targeted_structures=targeted_structures,
                                                        imaging_depths=imaging_depths,
                                                        transgenic_lines=transgenic_lines)

        if require_eye_tracking:
            experiments = [e for e in experiments
                           if e.get('fail_eye_tracking', None) is False]
        if not include_failed:
            experiments = [e for e in experiments 
                           if not e.get('experiment_container',{}).get('failed', False)]

        if experiment_container_ids is not None:
            experiments = [e for e in experiments if e[
                'experiment_container_id'] in experiment_container_ids]

        if session_types is not None:
            experiments = [e for e in experiments if e[
                'stimulus_name'] in session_types]

        if stimuli is not None:
            experiments = [e for e in experiments
                           if len(set(stimuli) & set(stimulus_info.stimuli_in_session(e['stimulus_name']))) > 0]

        return experiments

    def filter_cell_specimens(self, cell_specimens,
                              ids=None,
                              experiment_container_ids=None,
                              include_failed=False,
                              filters=None):
        """
        Filter a list of cell specimen records returned from the get_cell_metrics method according 
        some of their properties.

        Parameters
        ----------
        cell_specimens: list of dicts
            List of records returned by the get_cell_metrics method.

        ids: list of integers
            Return only records for cells with cell specimen ids in this list

        experiment_container_ids: list of integers
            Return only records for cells that belong to experiment container ids in this list

        include_failed: bool
            Whether to include cells from failed experiment containers

        filters: list of dicts
            Custom query used to reproduce filter sets created in the Allen Brain Observatory
            web application.  The general form is a list of dictionaries each of which
            describes a filtering operation based on a metric.  For more information, see
            dataframe_query.  
        """

        if not include_failed:
            cell_specimens = [c for c in cell_specimens if not c.get(
                    'failed_experiment_container', False)]

        if ids is not None:
            cell_specimens = [c for c in cell_specimens if c[
                'cell_specimen_id'] in ids]

        if experiment_container_ids is not None:
            cell_specimens = [c for c in cell_specimens if c[
                'experiment_container_id'] in experiment_container_ids]

        if filters is not None:
            cell_specimens = self.dataframe_query(cell_specimens,
                                                  filters,
                                                  'cell_specimen_id')

        return cell_specimens

    def dataframe_query_string(self,
                               filters):
        """
        Convert a list of cell metric filter dictionaries into a 
        Pandas query string.
        """

        def _quote_string(v):
            if isinstance(v, string_types):
                return "'%s'" % (v)
            else:
                return str(v)

        def _filter_clause(op, field, value):
            if op == 'in':
                query_args = [field, str(value)]
            elif type(value) is list:
                query_args = [field] + list(map(_quote_string, value))
            else:
                query_args = [field, str(value)]

            cluster_string = self._QUERY_TEMPLATES[op].\
                format(*query_args)

            return cluster_string

        query_string = ' & '.join(_filter_clause(f['op'],
                                                 f['field'],
                                                 f['value']) for f in filters)

        return query_string

    def dataframe_query(self,
                        data,
                        filters,
                        primary_key):
        """
        Given a list of dictionary records and a list of filter dictionaries,
        filter the records using Pandas and return the filtered set of records.
        
        Parameters
        ----------
        data: list of dicts
           List of dictionaries

        filters: list of dicts
           Each dictionary describes a filtering operation on a field in the dictionary.
           The general form is { 'field': <field>, 'op': <operation>, 'value': <filter_value(s)> }.
           For example, you can apply a threshold on the "osi_dg" column with something like this:
           { 'field': 'osi_dg', 'op': '>', 'value': 1.0 }.  See _QUERY_TEMPLATES for a full list
           of operators.
        """

        queries = self.dataframe_query_string(filters)
        result_dataframe = pd.DataFrame(data)
        result_dataframe = result_dataframe.query(queries)

        result_keys = set(result_dataframe[primary_key])
        result = [d for d in data
                  if d[primary_key]
                  in result_keys]

        return result

    def get_cell_specimen_id_mapping(self, file_name, mapping_table_id=None):
        '''Download mapping table from old to new cell specimen IDs.

        The mapping table is a CSV file that maps cell specimen ids
        that have changed between processing runs of the Brain
        Observatory pipeline.

        Parameters
        ----------
        file_name : string
            Filename to save locally.
        mapping_table_id : integer
            ID of the mapping table file. Defaults to the most recent
            mapping table. 

        Returns
        -------
        pandas.DataFrame
            Mapping table as a DataFrame.
        '''
        if mapping_table_id is None:
            mapping_table_id = self.CELL_MAPPING_ID
        data = self.template_query('brain_observatory_queries',
                                   'cell_specimen_id_mapping_table',
                                   mapping_table_id=mapping_table_id)

        try:
            file_url = data[0]['download_link']
        except Exception as _:
            raise Exception("No OphysCellSpecimenIdMapping file found.")

        self.retrieve_file_over_http(self.api_url + file_url, file_name)
        return pd.read_csv(file_name)
