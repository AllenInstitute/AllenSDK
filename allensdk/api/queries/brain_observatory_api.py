# Copyright 2016 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import os
from allensdk.api.queries.rma_template import RmaTemplate
from allensdk.config.manifest import Manifest
import allensdk.brain_observatory.stimulus_info as stimulus_info
import logging

class BrainObservatoryApi(RmaTemplate):
    _log = logging.getLogger('allensdk.api.queries.brain_observatory_api')

    NWB_FILE_TYPE = 'NWBOphys'

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
             'include': 'well_known_files(well_known_file_type),targeted_structure,specimen(donor(age,transgenic_lines))',
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
             'criteria_params': [ 'ophys_experiment_id' ]
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
             'include': 'ophys_experiments,isi_experiment,specimen(donor(age,transgenic_lines)),targeted_structure',
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
             'num_rows': 'all',
             'count': False,
             'criteria_params': ['cell_specimen_ids']
            }
        ]}
    
    
    def __init__(self, base_uri=None):
        super(BrainObservatoryApi, self).__init__(base_uri,
                                                  query_manifest=BrainObservatoryApi.rma_templates)
    
    
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
    
    
    # TODO: search by item type and level
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
    
    
    def get_cell_metrics(self, cell_specimen_ids=None):
        ''' Get cell metrics by id
        
        Parameters
        ----------
        cell_metrics_ids : integer or list of integers, optional
            only select specific cell metric records.
        
        Returns
        -------
        dict : cell metric metadata        
        '''
        self._log.warning("Downloading metrics and metadata for all cells. This can take some time.")

        data = self.template_query('brain_observatory_queries',
                                   'cell_metric',
                                   cell_specimen_ids=cell_specimen_ids)

        return data
    
    
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
    
    
    def save_ophys_experiment_data(self, ophys_experiment_id, file_name):
        dirname = os.path.dirname(file_name)
        Manifest.safe_mkdir(dirname)


        data = self.template_query('brain_observatory_queries',
                                   'ophys_experiment_data',
                                   ophys_experiment_id=ophys_experiment_id)
        
        try:
            file_url = data[0]['download_link']
        except Exception as _:
            raise Exception("ophys experiment %d has no data file" % ophys_experiment_id)

        self._log.warning("Downloading ophys_experiment %d NWB. This can take some time." % ophys_experiment_id)

        self.retrieve_file_over_http(self.api_url + file_url, file_name)
        
    
    def filter_experiment_containers(self, containers, 
                                     ids=None, 
                                     targeted_structures=None, 
                                     imaging_depths=None, 
                                     transgenic_lines=None):

        if ids is not None:
            containers = [ c for c in containers if c['id'] in ids ]

        if targeted_structures is not None:
            containers = [ c for c in containers if c['targeted_structure']['acronym'] in targeted_structures ]

        if imaging_depths is not None:
            containers = [ c for c in containers if c['imaging_depth'] in imaging_depths ]
        
        if transgenic_lines is not None:
            containers = [ c for c in containers for tl in c['specimen']['donor']['transgenic_lines'] if tl['name'] in transgenic_lines ]

        return containers
    
    
    def filter_ophys_experiments(self, experiments, 
                                 ids=None,
                                 experiment_container_ids=None,
                                 targeted_structures=None, 
                                 imaging_depths=None, 
                                 transgenic_lines=None, 
                                 stimuli=None, 
                                 session_types=None):

        # re-using the code from above
        experiments = self.filter_experiment_containers(experiments, 
                                                        ids=ids,
                                                        targeted_structures=targeted_structures, 
                                                        imaging_depths=imaging_depths, 
                                                        transgenic_lines=transgenic_lines)

        if experiment_container_ids is not None:
            experiments = [ e for e in experiments if e['experiment_container_id'] in experiment_container_ids ]
            
        if session_types is not None:
            experiments = [ e for e in experiments if e['stimulus_name'] in session_types ]

        if stimuli is not None:
            experiments = [ e for e in experiments 
                            if len(set(stimuli) & set(stimulus_info.stimuli_in_session(e['stimulus_name']))) > 0 ]

        return experiments

    def filter_cell_specimens(self, cell_specimens, ids=None, experiment_container_ids=None):
        if ids is not None:
            cell_specimens = [ c for c in cell_specimens if c['cell_specimen_id'] in ids ]

        if experiment_container_ids is not None:
            cell_specimens = [ c for c in cell_specimens if e['experiment_container_id'] in experiment_container_ids ]

        return cell_specimens
