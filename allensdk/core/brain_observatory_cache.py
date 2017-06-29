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
from . import json_utilities as ju
from allensdk.api.cache import Cache
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi
from allensdk.config.manifest_builder import ManifestBuilder
from .brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
import allensdk.brain_observatory.stimulus_info as stim_info
import six
from dateutil.parser import parse as parse_date



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
    MANIFEST_VERSION=None

    def __init__(self, cache=True, manifest_file='brain_observatory_manifest.json', base_uri=None):
        super(BrainObservatoryCache, self).__init__(
            manifest=manifest_file, cache=cache, version=self.MANIFEST_VERSION)
        self.api = BrainObservatoryApi(base_uri=base_uri)

    def get_all_targeted_structures(self):
        """ Return a list of all targeted structures in the data set. """
        containers = self.get_experiment_containers(simple=False)
        targeted_structures = set(
            [c['targeted_structure']['acronym'] for c in containers])
        return sorted(list(targeted_structures))

    def get_all_cre_lines(self):
        """ Return a list of all cre driver lines in the data set. """
        containers = self.get_experiment_containers(simple=False)
        cre_lines = set([_find_specimen_cre_line(c['specimen'])
                         for c in containers])
        return sorted(list(cre_lines))

    def get_all_reporter_lines(self):
        """ Return a list of all reporter lines in the data set. """
        containers = self.get_experiment_containers(simple=False)
        reporter_lines = set([_find_specimen_reporter_line(c['specimen'])
                              for c in containers])
        return sorted(list(reporter_lines))

    def get_all_imaging_depths(self):
        """ Return a list of all imaging depths in the data set. """
        containers = self.get_experiment_containers(simple=False)
        imaging_depths = set([c['imaging_depth'] for c in containers])
        return sorted(list(imaging_depths))

    def get_all_session_types(self):
        """ Return a list of all stimulus sessions in the data set. """
        exps = self.get_ophys_experiments(simple=False)
        names = set([exp['stimulus_name'] for exp in exps])
        return sorted(list(names))

    def get_all_stimuli(self):
        """ Return a list of all stimuli in the data set. """
        return sorted(list(stim_info.all_stimuli()))

    def get_experiment_containers(self, file_name=None,
                                  ids=None,
                                  targeted_structures=None,
                                  imaging_depths=None,
                                  cre_lines=None,
                                  transgenic_lines=None,
                                  include_failed=False,
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

        transgenic_lines: list
            List of transgenic lines. Must be in the list returned by
            BrainObservatoryCache.get_all_cre_lines() or.
            BrainObservatoryCache.get_all_reporter_lines().

        include_failed: boolean
            Whether or not to include failed experiment containers.

        simple: boolean
            Whether or not to simplify the dictionary properties returned by this method
            to a more concise subset.

        Returns
        -------
        list of dictionaries
        """
        _assert_not_string(targeted_structures, "targeted_structures")
        _assert_not_string(cre_lines, "cre_lines")
        _assert_not_string(transgenic_lines, "transgenic_lines")

        file_name = self.get_cache_path(
            file_name, self.EXPERIMENT_CONTAINERS_KEY)

        containers = self.api.get_experiment_containers(path=file_name,
                                                        strategy='lazy',
                                                        **Cache.cache_json())

        transgenic_lines = _merge_transgenic_lines(cre_lines, transgenic_lines)

        containers = self.api.filter_experiment_containers(containers, ids=ids,
                                                           targeted_structures=targeted_structures,
                                                           imaging_depths=imaging_depths,
                                                           transgenic_lines=transgenic_lines,
                                                           include_failed=include_failed)

        if simple:
            containers = [{
                'id': c['id'],
                'imaging_depth': c['imaging_depth'],
                'targeted_structure': c['targeted_structure']['acronym'],
                'cre_line': _find_specimen_cre_line(c['specimen']),
                'reporter_line': _find_specimen_reporter_line(c['specimen']),
                'donor_name': c['specimen']['donor']['external_donor_name'],
                'specimen_name': c['specimen']['name'],
                'tags': _find_container_tags(c),
                'failed': c['failed']
            } for c in containers]

        return containers

    def get_ophys_experiment_stimuli(self, experiment_id):
        """ For a single experiment, return the list of stimuli present in that experiment. """
        exps = self.get_ophys_experiments(ids=[experiment_id])

        if len(exps) == 0:
            return None

        return stim_info.stimuli_in_session(exps[0]['session_type'])

        
        
    def get_ophys_experiments(self, file_name=None,
                              ids=None,
                              experiment_container_ids=None,
                              targeted_structures=None,
                              imaging_depths=None,
                              cre_lines=None,
                              transgenic_lines=None,
                              stimuli=None,
                              session_types=None,
                              cell_specimen_ids=None,
                              include_failed=False,
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

        transgenic_lines: list
            List of transgenic lines. Must be in the list returned by
            BrainObservatoryCache.get_all_cre_lines() or.
            BrainObservatoryCache.get_all_reporter_lines().

        stimuli: list
            List of stimulus names.  Must be in the list returned by
            BrainObservatoryCache.get_all_stimuli().

        session_types: list
            List of stimulus session type names.  Must be in the list returned by
            BrainObservatoryCache.get_all_session_types().

        cell_specimen_ids: list
            Only include experiments that contain cells with these ids.

        include_failed: boolean
            Whether or not to include experiments from failed experiment containers.

        simple: boolean
            Whether or not to simplify the dictionary properties returned by this method
            to a more concise subset.

        Returns
        -------
        list of dictionaries
        """
        _assert_not_string(targeted_structures, "targeted_structures")
        _assert_not_string(cre_lines, "cre_lines")
        _assert_not_string(transgenic_lines, "transgenic_lines")
        _assert_not_string(stimuli, "stimuli")
        _assert_not_string(session_types, "session_types")

        file_name = self.get_cache_path(file_name, self.EXPERIMENTS_KEY)
        
        exps = self.api.get_ophys_experiments(path=file_name, 
                                              strategy='lazy', 
                                              **Cache.cache_json())

        transgenic_lines = _merge_transgenic_lines(cre_lines, transgenic_lines)

        if cell_specimen_ids is not None:
            cells = self.get_cell_specimens(ids=cell_specimen_ids)
            cell_container_ids = set([cell['experiment_container_id'] for cell in cells])
            if experiment_container_ids is not None:
                experiment_container_ids = list(set(experiment_container_ids) - cell_container_ids)
            else:
                experiment_container_ids = list(cell_container_ids)

        exps = self.api.filter_ophys_experiments(exps,
                                                 ids=ids,
                                                 experiment_container_ids=experiment_container_ids,
                                                 targeted_structures=targeted_structures,
                                                 imaging_depths=imaging_depths,
                                                 transgenic_lines=transgenic_lines,
                                                 stimuli=stimuli,
                                                 session_types=session_types,
                                                 include_failed=include_failed)

        if simple:
            exps = [{
                    'id': e['id'],
                    'imaging_depth': e['imaging_depth'],
                    'targeted_structure': e['targeted_structure']['acronym'],
                    'cre_line': _find_specimen_cre_line(e['specimen']),
                    'reporter_line': _find_specimen_reporter_line(e['specimen']),
                    'acquisition_age_days': _find_experiment_acquisition_age(e),
                    'experiment_container_id': e['experiment_container_id'],
                    'session_type': e['stimulus_name'],
                    'donor_name': e['specimen']['donor']['external_donor_name'],
                    'specimen_name': e['specimen']['name']
                    } for e in exps]
            
        return exps

    def _get_stimulus_mappings(self, file_name=None):
        """ Returns a mapping of which metrics are related to which stimuli. Internal use only. """

        file_name = self.get_cache_path(file_name, self.STIMULUS_MAPPINGS_KEY)

        mappings = self.api.get_stimulus_mappings(path=file_name,
                                                  strategy='lazy',
                                                  **Cache.cache_json())

        return mappings

    def get_cell_specimens(self,
                           file_name=None,
                           ids=None,
                           experiment_container_ids=None,
                           include_failed=False,
                           simple=True,
                           filters=None):
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

        include_failed: bool
            Whether to include cells from failed experiment containers

        simple: boolean
            Whether or not to simplify the dictionary properties returned by this method
            to a more concise subset.

        filters: list of dicts
            List of filter dictionaries.  The Allen Brain Observatory web site can 
            generate filters in this format to reproduce a filtered set of cells
            found there.  To see what these look like, visit 
            http://observatory.brain-map.org/visualcoding, perform a cell search
            and apply some filters (e.g. find cells in a particular area), then 
            click the "view these cells in the AllenSDK" link on the bottom-left
            of the search results page.  This will take you to a page that contains
            a code sample you can use to apply those same filters via this argument.
            For more detail on the filter syntax, see BrainObservatoryApi.dataframe_query.
            

        Returns
        -------
        list of dictionaries
        """

        file_name = self.get_cache_path(file_name, self.CELL_SPECIMENS_KEY)

        cell_specimens = self.api.get_cell_metrics(path=file_name,
                                                   strategy='lazy',
                                                   **Cache.cache_json())

        cell_specimens = self.api.filter_cell_specimens(cell_specimens,
                                                        ids=ids,
                                                        experiment_container_ids=experiment_container_ids,
                                                        include_failed=include_failed,
                                                        filters=filters)

        # drop the thumbnail columns
        if simple:
            mappings = self._get_stimulus_mappings()
            thumbnails = [m['item'] for m in mappings if m[
                'item_type'] == 'T' and m['level'] == 'R']
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
        file_name = self.get_cache_path(
            file_name, self.EXPERIMENT_DATA_KEY, ophys_experiment_id)

        self.api.save_ophys_experiment_data(ophys_experiment_id, file_name, strategy='lazy')

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
        mb.set_version(self.MANIFEST_VERSION)
        mb.add_path('BASEDIR', '.')
        mb.add_path(self.EXPERIMENT_CONTAINERS_KEY,
                    'experiment_containers.json', typename='file', parent_key='BASEDIR')
        mb.add_path(self.EXPERIMENTS_KEY, 'ophys_experiments.json',
                    typename='file', parent_key='BASEDIR')
        mb.add_path(self.EXPERIMENT_DATA_KEY, 'ophys_experiment_data/%d.nwb',
                    typename='file', parent_key='BASEDIR')
        mb.add_path(self.CELL_SPECIMENS_KEY, 'cell_specimens.json',
                    typename='file', parent_key='BASEDIR')
        mb.add_path(self.STIMULUS_MAPPINGS_KEY, 'stimulus_mappings.json',
                    typename='file', parent_key='BASEDIR')

        mb.write_json_file(file_name)


def _find_specimen_cre_line(specimen):
    try:
        return next(tl['name'] for tl in specimen['donor']['transgenic_lines']
                    if tl['transgenic_line_type_name'] == 'driver' and
                    'Cre' in tl['name'])
    except StopIteration:
        return None

def _find_specimen_reporter_line(specimen):
    try:
        return next(tl['name'] for tl in specimen['donor']['transgenic_lines']
                    if tl['transgenic_line_type_name'] == 'reporter')
    except StopIteration:
        return None


def _find_experiment_acquisition_age(exp):
    try:
        return (parse_date(exp['date_of_acquisition']) - parse_date(exp['specimen']['donor']['date_of_birth'])).days
    except KeyError as e:
        return None


def _merge_transgenic_lines(*lines_list):
    transgenic_lines = set()

    for lines in lines_list:
        if lines is not None:
            for line in lines:
                transgenic_lines.add(line)

    if len(transgenic_lines):
        return list(transgenic_lines)
    else:
        return None

def _find_container_tags(container):
    """ Custom logic for extracting tags from donor conditions.  Filtering 
    out tissuecyte tags. """
    conditions = container['specimen']['donor'].get('conditions', [])
    return [c['name'] for c in conditions if not c['name'].startswith('tissuecyte')]

def _assert_not_string(arg, name):
    if isinstance(arg, six.string_types):
        raise TypeError(
            "Argument '%s' with value '%s' is a string type, but should be a list." % (name, arg))
