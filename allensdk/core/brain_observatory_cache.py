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
import allensdk.core.json_utilities as ju
from allensdk.api.cache import Cache
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi
from allensdk.config.manifest_builder import ManifestBuilder
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
import allensdk.brain_observatory.stimulus_info as stim_info
import pandas as pd


class BrainObservatoryCache(Cache):
    """
    Cache class for storing and accessing data from the Brain Observatory.
    By default, this class will cache any downloaded metadata or files in 
    well known locations defined in a manifest file.  This behavior can be
    disabled.

    Attributes
    ----------
    
    api: BrainObservatoryApi instance
        The object used for making API queries related to the Brain

 Observatory.

    Parameters
    ----------
    
    cache: boolean
        Whether the class should save results of API queries to locations specified
        in the manifest file.  Queries for files (as opposed to metadata) must have a
        file location.  If caching is disabled, those locations must be specified
        in the function call (e.g. get_ophys_experiment_data(file_name='file.nwb')).

    manifest_file: string
       File name of the manifest to be read.  Default is "brain_observatory_manifest.json".
    """

    EXPERIMENT_CONTAINERS_KEY = 'EXPERIMENT_CONTAINERS'
    EXPERIMENTS_KEY = 'EXPERIMENTS'
    CELL_SPECIMENS_KEY = 'CELL_SPECIMENS'
    EXPERIMENT_DATA_KEY = 'EXPERIMENT_DATA'
    STIMULUS_MAPPINGS_KEY = 'STIMULUS_MAPPINGS'
    
    def __init__(self, cache=True, manifest_file='brain_observatory_manifest.json', base_uri=None):
        super(BrainObservatoryCache, self).__init__(manifest=manifest_file, cache=cache)
        self.api = BrainObservatoryApi(base_uri=base_uri)
        

    def get_all_targeted_structures(self):
        """ Return a list of all targeted structures in the data set. """
        containers = self.get_experiment_containers(simple=False)
        targeted_structures = set([ c['targeted_structure']['acronym'] for c in containers])
        return sorted(list(targeted_structures))


    def get_all_cre_lines(self):
        """ Return a list of all cre driver lines in the data set. """
        containers = self.get_experiment_containers(simple=False)
        cre_lines = set([ _find_specimen_cre_line(c['specimen']) for c in containers ])
        return sorted(list(cre_lines))


    def get_all_imaging_depths(self):
        """ Return a list of all imaging depths in the data set. """
        containers = self.get_experiment_containers(simple=False)
        imaging_depths = set([ c['imaging_depth'] for c in containers ])
        return sorted(list(imaging_depths))


    def get_all_session_types(self):
        """ Return a list of all stimulus sessions in the data set. """
        exps = self.get_ophys_experiments()
        names = set([ exp['stimulus_name'] for exp in exps ])
        return sorted(list(names))


    def get_all_stimuli(self):
        """ Return a list of all stimuli in the data set. """
        return sorted(list(stim_info.all_stimuli()))
    

    def get_experiment_containers(self, file_name=None, 
                                  ids=None, 
                                  targeted_structures=None, 
                                  imaging_depths=None, 
                                  cre_lines=None, 
                                  simple=True):
        """ Get a list of experiment containers matching certain criteria.
        
        Parameters
        ----------
        file_name: string
            File name to save/read the experiment containers.  If file_name is None, 
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.

        ids: list
            List of experiment container ids.  

        targeted_structures: list
            List of structure acronyms.  Must be in the list returned by 
            BrainObservatoryCache.get_all_targeted_structures().

        imaging_depths: list
            List of imaging depths.  Must be in the list returned by 
            BrainObservatoryCache.get_all_imaging_depths().

        cre_lines: list
            List of cre lines.  Must be in the list returned by 
            BrainObservatoryCache.get_all_cre_lines().

        simple: boolean
            Whether or not to simplify the dictionary properties returned by this method
            to a more concise subset.

        Returns
        -------
        list of dictionaries
        """
        file_name = self.get_cache_path(file_name, self.EXPERIMENT_CONTAINERS_KEY)

        if os.path.exists(file_name):
            containers = ju.read(file_name)
        else:
            containers = self.api.get_experiment_containers()

            if self.cache:
                ju.write(file_name, containers)

        containers = self.api.filter_experiment_containers(containers, ids=ids,
                                                           targeted_structures=targeted_structures, 
                                                           imaging_depths=imaging_depths, 
                                                           transgenic_lines=cre_lines)

        if simple:
            containers = [ {
                    'id': c['id'],
                    'imaging_depth': c['imaging_depth'],
                    'targeted_structure': c['targeted_structure']['acronym'],
                    'cre_line': _find_specimen_cre_line(c['specimen']),
                    'age_days': c['specimen']['donor']['age']['days']
                    } for c in containers ]

        return containers


    def get_ophys_experiments(self, file_name=None, 
                              ids=None,
                              experiment_container_ids=None,
                              targeted_structures=None, 
                              imaging_depths=None, 
                              cre_lines=None,
                              stimuli=None, 
                              session_types=None, 
                              simple=True):
        """ Get a list of ophys experiments matching certain criteria.
        
        Parameters
        ----------
        file_name: string
            File name to save/read the ophys experiments.  If file_name is None, 
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.

        ids: list
            List of ophys experiment ids.  

        experiment_container_ids: list
            List of experiment container ids.

        targeted_structures: list
            List of structure acronyms.  Must be in the list returned by 
            BrainObservatoryCache.get_all_targeted_structures().

        imaging_depths: list
            List of imaging depths.  Must be in the list returned by 
            BrainObservatoryCache.get_all_imaging_depths().

        cre_lines: list
            List of cre lines.  Must be in the list returned by 
            BrainObservatoryCache.get_all_cre_lines().

        stimuli: list
            List of stimulus names.  Must be in the list returned by 
            BrainObservatoryCache.get_all_stimuli().

        session_types: list
            List of stimulus session type names.  Must be in the list returned by 
            BrainObservatoryCache.get_all_session_types().

        simple: boolean
            Whether or not to simplify the dictionary properties returned by this method
            to a more concise subset.

        Returns
        -------
        list of dictionaries
        """
        file_name = self.get_cache_path(file_name, self.EXPERIMENTS_KEY)

        if os.path.exists(file_name):
            exps = ju.read(file_name)
        else:
            exps = self.api.get_ophys_experiments()

            if self.cache:
                ju.write(file_name, exps)

        exps = self.api.filter_ophys_experiments(exps, 
                                                 ids=ids,
                                                 experiment_container_ids=experiment_container_ids, 
                                                 targeted_structures=targeted_structures, 
                                                 imaging_depths=imaging_depths, 
                                                 transgenic_lines=cre_lines, 
                                                 stimuli=stimuli, 
                                                 session_types=session_types)

        if simple:
            exps = [ {
                    'id': e['id'],
                    'imaging_depth': e['imaging_depth'],
                    'targeted_structure': e['targeted_structure']['acronym'],
                    'cre_line': _find_specimen_cre_line(e['specimen']),
                    'age_days': e['specimen']['donor']['age']['days'],
                    'experiment_container_id': e['experiment_container_id'],
                    'session_type': e['stimulus_name']
                    } for e in exps ]
        return exps


    def _get_stimulus_mappings(self, file_name=None):
        """ Returns a mapping of which metrics are related to which stimuli. Internal use only. """

        file_name = self.get_cache_path(file_name, self.STIMULUS_MAPPINGS_KEY)

        if os.path.exists(file_name):
            mappings = ju.read(file_name)
        else:
            mappings = self.api.get_stimulus_mappings()

            if self.cache:
                ju.write(file_name, mappings)

        return mappings


    def get_cell_specimens(self, file_name=None, ids=None, experiment_container_ids=None, simple=True):
        """ Return cell specimens that have certain properies.
        
        Parameters
        ----------
        file_name: string
            File name to save/read the cell specimens.  If file_name is None, 
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.

        ids: list
            List of cell specimen ids.  

        experiment_container_ids: list
            List of experiment container ids.
            
        simple: boolean
            Whether or not to simplify the dictionary properties returned by this method
            to a more concise subset.
            
        Returns
        -------
        list of dictionaries
        """

        file_name = self.get_cache_path(file_name, self.CELL_SPECIMENS_KEY)

        if os.path.exists(file_name):
            cell_specimens = ju.read(file_name)
        else:
            cell_specimens = self.api.get_cell_metrics()

            if self.cache:
                ju.write(file_name, cell_specimens)

        cell_specimens = self.api.filter_cell_specimens(cell_specimens, 
                                                        ids=ids, 
                                                        experiment_container_ids=experiment_container_ids)
        
        # drop the thumbnail columns
        if simple:
            mappings = self._get_stimulus_mappings()
            thumbnails = [ m['item'] for m in mappings if m['item_type'] == 'T' and m['level'] == 'R']
            for cs in cell_specimens:
                for t in thumbnails:
                    del cs[t]
                
        return cell_specimens

    
    def get_ophys_experiment_data(self, ophys_experiment_id, file_name=None):
        """ Download the NWB file for an ophys_experiment (if it hasn't already been
        downloaded) and return a data accessor object.

        Parameters
        ----------
        file_name: string
            File name to save/read the data set.  If file_name is None, 
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.        

        ophys_experiment_id: integer
            id of the ophys_experiment to retrieve
            
        Returns
        -------
        BrainObservatoryNwbDataSet
        """
        file_name = self.get_cache_path(file_name, self.EXPERIMENT_DATA_KEY, ophys_experiment_id)

        if not os.path.exists(file_name):
            self.api.save_ophys_experiment_data(ophys_experiment_id, file_name)

        return BrainObservatoryNwbDataSet(file_name)

    
    def build_manifest(self, file_name):
        """
        Construct a manifest for this Cache class and save it in a file.
        
        Parameters
        ----------
        
        file_name: string
            File location to save the manifest.

        """

        mb = ManifestBuilder()

        mb.add_path('BASEDIR', '.')
        mb.add_path(self.EXPERIMENT_CONTAINERS_KEY, 'experiment_containers.json', typename='file', parent_key='BASEDIR')
        mb.add_path(self.EXPERIMENTS_KEY, 'ophys_experiments.json', typename='file', parent_key='BASEDIR')
        mb.add_path(self.EXPERIMENT_DATA_KEY, 'ophys_experiment_data/%d.nwb', typename='file', parent_key='BASEDIR')
        mb.add_path(self.CELL_SPECIMENS_KEY, 'cell_specimens.json', typename='file', parent_key='BASEDIR')
        mb.add_path(self.STIMULUS_MAPPINGS_KEY, 'stimulus_mappings.json', typename='file', parent_key='BASEDIR')

        mb.write_json_file(file_name)

def _find_specimen_cre_line(specimen):
    return next(tl['name'] for tl in specimen['donor']['transgenic_lines'] if 'Cre' in tl['name'])
