# Copyright 2015-2016 Allen Institute for Brain Science
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

from allensdk.config.manifest_builder import ManifestBuilder
from allensdk.api.cache import Cache
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.deprecated import deprecated

from . import json_utilities
from .ontology import Ontology
from .structure_tree import StructureTree
from .reference_space import ReferenceSpace

import nrrd
import os
import pandas as pd
import numpy as np
from allensdk.config.manifest import Manifest
import warnings
import operator as op
import functools


class MouseConnectivityCache(Cache):
    """
    Cache class for storing and accessing data related to the adult mouse
    Connectivity Atlas.  By default, this class will cache any downloaded
    metadata or files in well known locations defined in a manifest file.
    This behavior can be disabled.

    Attributes
    ----------

    resolution: int
        Resolution of grid data to be downloaded when accessing projection volume,
        the annotation volume, and the annotation volume.  Must be one of (10, 25,
        50, 100).  Default is 25.

    api: MouseConnectivityApi instance
        Used internally to make API queries.

    Parameters
    ----------

    resolution: int
        Resolution of grid data to be downloaded when accessing projection volume,
        the annotation volume, and the annotation volume.  Must be one of (10, 25,
        50, 100).  Default is 25.

    ccf_version: string
        Desired version of the Common Coordinate Framework.  This affects the annotation 
        volume (get_annotation_volume) and structure masks (get_structure_mask). 
        Must be one of (MouseConnectivityApi.CCF_2015, MouseConnectivityApi.CCF_2016). 
        Default: MouseConnectivityApi.CCF_2016

    cache: boolean
        Whether the class should save results of API queries to locations specified
        in the manifest file.  Queries for files (as opposed to metadata) must have a
        file location.  If caching is disabled, those locations must be specified
        in the function call (e.g. get_projection_density(file_name='file.nrrd')).

    manifest_file: string
        File name of the manifest to be read.  Default is "mouse_connectivity_manifest.json".

    """

    CCF_VERSION_KEY = 'CCF_VERSION'
    ANNOTATION_KEY = 'ANNOTATION'
    TEMPLATE_KEY = 'TEMPLATE'
    PROJECTION_DENSITY_KEY = 'PROJECTION_DENSITY'
    INJECTION_DENSITY_KEY = 'INJECTION_DENSITY'
    INJECTION_FRACTION_KEY = 'INJECTION_FRACTION'
    DATA_MASK_KEY = 'DATA_MASK'
    STRUCTURE_UNIONIZES_KEY = 'STRUCTURE_UNIONIZES'
    EXPERIMENTS_KEY = 'EXPERIMENTS'
    STRUCTURES_KEY = 'STRUCTURES'
    STRUCTURE_TREE_KEY = 'STRUCTURE_TREE'
    STRUCTURE_MASK_KEY = 'STRUCTURE_MASK'
    MANIFEST_VERSION = 1.0

    def __init__(self,
                 resolution=None,
                 cache=True,
                 manifest_file='mouse_connectivity_manifest.json',
                 ccf_version=None,
                 base_uri=None):
        super(MouseConnectivityCache, self).__init__(
            manifest=manifest_file, cache=cache, version=self.MANIFEST_VERSION)

        if resolution is None:
            self.resolution = MouseConnectivityApi.VOXEL_RESOLUTION_25_MICRONS
        else:
            self.resolution = resolution
        self.api = MouseConnectivityApi(base_uri=base_uri)

        if ccf_version is None:
            ccf_version = MouseConnectivityApi.CCF_VERSION_DEFAULT
        self.ccf_version = ccf_version
        
    def get_annotation_volume(self, file_name=None):
        """
        Read the annotation volume.  Download it first if it doesn't exist.

        Parameters
        ----------

        file_name: string
            File name to store the annotation volume.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        """

        file_name = self.get_cache_path(
            file_name, self.ANNOTATION_KEY, self.ccf_version, self.resolution)

        annotation, info = self.api.download_annotation_volume(
            self.ccf_version,
            self.resolution,
            file_name, 
            strategy='lazy')

        return annotation, info

    def get_template_volume(self, file_name=None):
        """
        Read the template volume.  Download it first if it doesn't exist.

        Parameters
        ----------

        file_name: string
            File name to store the template volume.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        """

        file_name = self.get_cache_path(
            file_name, self.TEMPLATE_KEY, self.resolution)

        template, info = self.api.download_template_volume(self.resolution, 
                                                           file_name, 
                                                           strategy='lazy')

        return template, info

    def get_projection_density(self, experiment_id, file_name=None):
        """
        Read a projection density volume for a single experiment.  Download it
        first if it doesn't exist.  Projection density is the proportion of
        of projecting pixels in a grid voxel in [0,1].

        Parameters
        ----------

        experiment_id: int
            ID of the experiment to download/read.  This corresponds to
            section_data_set_id in the API.

        file_name: string
            File name to store the template volume.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        """
        
        file_name = self.get_cache_path(file_name,
                                        self.PROJECTION_DENSITY_KEY,
                                        experiment_id,
                                        self.resolution)
                                        
        self.api.download_projection_density(
            file_name, experiment_id, self.resolution, strategy='lazy')

        return nrrd.read(file_name)

    def get_injection_density(self, experiment_id, file_name=None):
        """
        Read an injection density volume for a single experiment. Download it
        first if it doesn't exist.  Injection density is the proportion of
        projecting pixels in a grid voxel only including pixels that are
        part of the injection site in [0,1].

        Parameters
        ----------

        experiment_id: int
            ID of the experiment to download/read.  This corresponds to
            section_data_set_id in the API.

        file_name: string
            File name to store the template volume.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        """
        
        file_name = self.get_cache_path(file_name,
                                        self.INJECTION_DENSITY_KEY,
                                        experiment_id,
                                        self.resolution)
        self.api.download_injection_density(
            file_name, experiment_id, self.resolution, strategy='lazy')

        return nrrd.read(file_name)

    def get_injection_fraction(self, experiment_id, file_name=None):
        """
        Read an injection fraction volume for a single experiment. Download it
        first if it doesn't exist.  Injection fraction is the proportion of
        pixels in the injection site in a grid voxel in [0,1].

        Parameters
        ----------

        experiment_id: int
            ID of the experiment to download/read.  This corresponds to
            section_data_set_id in the API.

        file_name: string
            File name to store the template volume.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        """

        file_name = self.get_cache_path(file_name,
                                        self.INJECTION_FRACTION_KEY,
                                        experiment_id,
                                        self.resolution)
        self.api.download_injection_fraction(
            file_name, experiment_id, self.resolution, strategy='lazy')

        return nrrd.read(file_name)

    def get_data_mask(self, experiment_id, file_name=None):
        """
        Read a data mask volume for a single experiment. Download it
        first if it doesn't exist.  Data mask is a binary mask of
        voxels that have valid data.  Only use valid data in analysis!

        Parameters
        ----------

        experiment_id: int
            ID of the experiment to download/read.  This corresponds to
            section_data_set_id in the API.

        file_name: string
            File name to store the template volume.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        """

        file_name = self.get_cache_path(file_name,
                                        self.DATA_MASK_KEY,
                                        experiment_id,
                                        self.resolution)
        self.api.download_data_mask(
            file_name, experiment_id, self.resolution, strategy='lazy')

        return nrrd.read(file_name)

    def get_structure_tree(self, file_name=None):
        """
        Read the list of adult mouse structures and return an StructureTree 
        instance.

        Parameters
        ----------

        file_name: string
            File name to save/read the structures table.  If file_name is None,
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.
        """
        
        file_name = self.get_cache_path(file_name, self.STRUCTURE_TREE_KEY)

        return OntologiesApi(self.api.api_url).get_structures_with_sets(
            strategy='lazy',
            path=file_name,
            pre=StructureTree.clean_structures, 
            post=StructureTree, 
            structure_graph_ids=1,
            **Cache.cache_json())

    @deprecated('Use get_structure_tree instead.')
    def get_ontology(self, file_name=None):
        """
        Read the list of adult mouse structures and return an Ontology instance.

        Parameters
        ----------

        file_name: string
            File name to save/read the structures table.  If file_name is None,
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.
        """

        return Ontology(self.get_structures(file_name))

    @deprecated('Use get_structure_tree instead.')
    def get_structures(self, file_name=None):
        """
        Read the list of adult mouse structures and return a Pandas DataFrame.

        Parameters
        ----------

        file_name: string
            File name to save/read the structures table.  If file_name is None,
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.
        """
        file_name = self.get_cache_path(file_name, self.STRUCTURES_KEY)

        return OntologiesApi(base_uri=self.api.api_url).get_structures(
            1,
            strategy='lazy',
            path=file_name,
            **Cache.cache_csv_dataframe())


    def get_experiments(self, dataframe=False, file_name=None, cre=None, injection_structure_ids=None):
        """
        Read a list of experiments that match certain criteria.  If caching is enabled,
        this will save the whole (unfiltered) list of experiments to a file.

        Parameters
        ----------

        dataframe: boolean
            Return the list of experiments as a Pandas DataFrame.  If False,
            return a list of dictionaries.  Default False.

        file_name: string
            File name to save/read the structures table.  If file_name is None,
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.

        cre: boolean or list
            If True, return only cre-positive experiments.  If False, return only
            cre-negative experiments.  If None, return all experients. If list, return
            all experiments with cre line names in the supplied list. Default None.

        injection_structure_ids: list
            Only return experiments that were injected in the structures provided here.
            If None, return all experiments.  Default None.

        """

        file_name = self.get_cache_path(file_name, self.EXPERIMENTS_KEY)

        if os.path.exists(file_name):
            experiments = json_utilities.read(file_name)
        else:
            experiments = self.api.experiment_source_search(
                injection_structures='root')

            # removing these elements because they are specific to a particular
            # resolution
            for e in experiments:
                del e['num-voxels']
                del e['injection-volume']
                del e['sum']
                del e['name']

            if self.cache:
                Manifest.safe_make_parent_dirs(file_name)

                json_utilities.write(file_name, experiments)

        # filter the read/downloaded list of experiments
        experiments = self.filter_experiments(
            experiments, cre, injection_structure_ids)

        if dataframe:
            experiments = pd.DataFrame(experiments)
            experiments.set_index(['id'], inplace=True, drop=False)

        return experiments

    def filter_experiments(self, experiments, cre=None, injection_structure_ids=None):
        """
        Take a list of experiments and filter them by cre status and injection structure.

        Parameters
        ----------

        cre: boolean or list
            If True, return only cre-positive experiments.  If False, return only
            cre-negative experiments.  If None, return all experients. If list, return
            all experiments with cre line names in the supplied list. Default None.

        injection_structure_ids: list
            Only return experiments that were injected in the structures provided here.
            If None, return all experiments.  Default None.
        """

        if cre is True:
            experiments = [e for e in experiments if e['transgenic-line']]
        elif cre is False:
            experiments = [e for e in experiments if not e['transgenic-line']]
        elif cre is not None:
            experiments = [e for e in experiments if e[
                'transgenic-line'] in cre]

        if injection_structure_ids is not None:
            structure_ids = MouseConnectivityCache.validate_structure_ids(injection_structure_ids)

            descendant_ids = reduce(op.add, self.get_structure_tree()\
                                    .descendant_ids(injection_structure_ids))
            experiments = [e for e in experiments 
                           if e['structure-id'] in descendant_ids]

        return experiments

    def get_experiment_structure_unionizes(self, experiment_id, 
                                           file_name=None, 
                                           is_injection=None,
                                           structure_ids=None, 
                                           include_descendants=False,
                                           hemisphere_ids=None):
        """
        Retrieve the structure unionize data for a specific experiment.  Filter by
        structure, injection status, and hemisphere.

        Parameters
        ----------

        experiment_id: int
            ID of the experiment of interest.  Corresponds to section_data_set_id in the API.

        file_name: string
            File name to save/read the experiments list.  If file_name is None,
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.

        is_injection: boolean
            If True, only return unionize records that disregard non-injection pixels.
            If False, only return unionize records that disregard injection pixels.
            If None, return all records.  Default None.

        structure_ids: list
            Only return unionize records for a specific set of structures.
            If None, return all records. Default None.

        include_descendants: boolean
            Include all descendant records for specified structures. Default False.

        hemisphere_ids: list
            Only return unionize records that disregard pixels outside of a hemisphere.
            or set of hemispheres. Left = 1, Right = 2, Both = 3.  If None, include all
            records [1, 2, 3].  Default None.

        """

        file_name = self.get_cache_path(file_name, 
                                        self.STRUCTURE_UNIONIZES_KEY, 
                                        experiment_id)
                   
        filter_fn = functools.partial(self.filter_structure_unionizes, 
                                      is_injection=is_injection, 
                                      structure_ids=structure_ids, 
                                      include_descendants=include_descendants, 
                                      hemisphere_ids=hemisphere_ids)
                                      
        col_rn = lambda x: pd.DataFrame(x).rename(columns={
            'section_data_set_id': 'experiment_id'})
                                      
        return self.api.get_structure_unionizes([experiment_id], 
                                                path=file_name,
                                                strategy='lazy',
                                                pre=col_rn,
                                                post=filter_fn,
                                                writer=lambda p, x : pd.DataFrame(x).to_csv(p),
                                                reader=pd.DataFrame.from_csv)

    def filter_structure_unionizes(self, unionizes, 
                                   is_injection=None, 
                                   structure_ids=None, 
                                   include_descendants=False,
                                   hemisphere_ids=None):
        """
        Take a list of unionzes and return a subset of records filtered by injection status, structure, and
        hemisphere.

        Parameters
        ----------
        is_injection: boolean
            If True, only return unionize records that disregard non-injection pixels.
            If False, only return unionize records that disregard injection pixels.
            If None, return all records.  Default None.

        structure_ids: list
            Only return unionize records for a set of structures.
            If None, return all records. Default None.

        include_descendants: boolean
            Include all descendant records for specified structures. Default False.

        hemisphere_ids: list
            Only return unionize records that disregard pixels outside of a hemisphere.
            or set of hemispheres. Left = 1, Right = 2, Both = 3.  If None, include all
            records [1, 2, 3].  Default None.
        """
        if is_injection is not None:
            unionizes = unionizes[unionizes.is_injection == is_injection]

        if structure_ids is not None:
            structure_ids = MouseConnectivityCache.validate_structure_ids(structure_ids)

            if include_descendants:
                structure_ids = reduce(op.add, self.get_structure_tree().descendant_ids(structure_ids))
            else:
                structure_ids = set(structure_ids)


            unionizes = unionizes[
                unionizes['structure_id'].isin(structure_ids)]

        if hemisphere_ids is not None:
            unionizes = unionizes[
                unionizes['hemisphere_id'].isin(hemisphere_ids)]

        return unionizes

    def get_structure_unionizes(self, experiment_ids, 
                                is_injection=None, 
                                structure_ids=None, 
                                include_descendants=False,
                                hemisphere_ids=None):
        """
        Get structure unionizes for a set of experiment IDs.  Filter the results by injection status,
        structure, and hemisphere.

        Parameters
        ----------
        experiment_ids: list
            List of experiment IDs.  Corresponds to section_data_set_id in the API.

        is_injection: boolean
            If True, only return unionize records that disregard non-injection pixels.
            If False, only return unionize records that disregard injection pixels.
            If None, return all records.  Default None.

        structure_ids: list
            Only return unionize records for a specific set of structures.
            If None, return all records. Default None.

        include_descendants: boolean
            Include all descendant records for specified structures. Default False.

        hemisphere_ids: list
            Only return unionize records that disregard pixels outside of a hemisphere.
            or set of hemispheres. Left = 1, Right = 2, Both = 3.  If None, include all
            records [1, 2, 3].  Default None.
        """

        unionizes = [self.get_experiment_structure_unionizes(eid,
                                                             is_injection=is_injection,
                                                             structure_ids=structure_ids,
                                                             include_descendants=include_descendants,
                                                             hemisphere_ids=hemisphere_ids)
                     for eid in experiment_ids]

        return pd.concat(unionizes, ignore_index=True)

    def get_projection_matrix(self, experiment_ids, 
                              projection_structure_ids,
                              hemisphere_ids=None, 
                              parameter='projection_volume', 
                              dataframe=False):

        unionizes = self.get_structure_unionizes(experiment_ids,
                                                 is_injection=False,
                                                 structure_ids=projection_structure_ids,
                                                 include_descendants=False,
                                                 hemisphere_ids=hemisphere_ids)

        hemisphere_ids = set(unionizes['hemisphere_id'].values.tolist())

        nrows = len(experiment_ids)
        ncolumns = len(projection_structure_ids) * len(hemisphere_ids)

        matrix = np.empty((nrows, ncolumns))
        matrix[:] = np.NAN

        row_lookup = {}
        for idx, e in enumerate(experiment_ids):
            row_lookup[e] = idx

        column_lookup = {}
        columns = []

        cidx = 0
        hlabel = {1: '-L', 2: '-R', 3: ''}

        acronym_map = self.get_structure_tree().value_map(lambda x: x['id'], 
                                                          lambda x: x['acronym'])

        for hid in hemisphere_ids:
            for sid in projection_structure_ids:
                column_lookup[(hid, sid)] = cidx
                label = acronym_map[sid] + hlabel[hid]
                columns.append(
                    {'hemisphere_id': hid, 'structure_id': sid, 'label': label})
                cidx += 1

        for _, row in unionizes.iterrows():
            ridx = row_lookup[row['experiment_id']]
            k = (row['hemisphere_id'], row['structure_id'])
            cidx = column_lookup[k]
            matrix[ridx, cidx] = row[parameter]

        if dataframe:
            warnings.warn("dataframe argument is deprecated.")
            all_experiments = self.get_experiments(dataframe=True)

            rows_df = all_experiments.loc[experiment_ids]

            cols_df = pd.DataFrame(columns)

            return {'matrix': matrix, 'rows': rows_df, 'columns': cols_df}
        else:
            return {'matrix': matrix, 'rows': experiment_ids, 'columns': columns}

    def get_reference_space(self, structure_file_name=None, 
                            annotation_file_name=None):
        """
        Build a ReferenceSpace from this cache's annotation volume and 
        structure tree. The ReferenceSpace does operations that relate brain 
        structures to spatial domains.
        
        Parameters
        ----------
        
        structure_file_name: string
            File name to save/read the structures table.  If file_name is None,
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.
            
        annotation_file_name: string
            File name to store the annotation volume.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.
        
        """
        
        return ReferenceSpace(self.get_structure_tree(structure_file_name), 
                              self.get_annotation_volume(annotation_file_name)[0], 
                              [self.resolution] * 3)

    def get_structure_mask(self, structure_id, file_name=None, annotation_file_name=None):
        """
        Read a 3D numpy array shaped like the annotation volume that has non-zero values where
        voxels belong to a particular structure.  This will take care of identifying substructures.

        Notes
        -----
        If you are making large numbers of masks, there is a faster structure mask generator in 
        ReferenceSpace.many_structure_masks.  We will be migrating this function in a future release.
        
        Parameters
        ----------

        structure_id: int
            ID of a structure.

        file_name: string
            File name to store the structure mask.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.

        annotation_file_name: string
            File name to store the annotation volume.  If it already exists,
            it will be read from this file.  If file_name is None, the
            file_name will be pulled out of the manifest.  Default is None.
        """
        structure_id = MouseConnectivityCache.validate_structure_id(structure_id)

        file_name = self.get_cache_path(
            file_name, self.STRUCTURE_MASK_KEY, self.ccf_version, 
            self.resolution, structure_id)

        if os.path.exists(file_name):
            return nrrd.read(file_name)
        else:
            st = self.get_structure_tree()
            structure_ids = st.descendant_ids([structure_id])[0]
            annotation, _ = self.get_annotation_volume(annotation_file_name)
            mask = self.make_structure_mask(structure_ids, annotation)

            if self.cache:
                Manifest.safe_make_parent_dirs(file_name)
                nrrd.write(file_name, mask)

            return mask, None

    def make_structure_mask(self, structure_ids, annotation):
        """
        Look at an annotation volume and identify voxels that have values
        in a list of structure ids.

        Parameters
        ----------

        structure_ids: list
            List of IDs to look for in the annotation volume

        annotation: np.ndarray
            Numpy array filled with IDs.

        """

        m = np.zeros(annotation.shape, dtype=np.uint8)

        for _, sid in enumerate(structure_ids):
            m[annotation == sid] = 1

        return m

    def build_manifest(self, file_name):
        """
        Construct a manifest for this Cache class and save it in a file.

        Parameters
        ----------

        file_name: string
            File location to save the manifest.

        """

        manifest_builder = ManifestBuilder()
        manifest_builder.set_version(self.MANIFEST_VERSION)
        manifest_builder.add_path('BASEDIR', '.')

        manifest_builder.add_path(self.EXPERIMENTS_KEY,
                                  'experiments.json',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.STRUCTURES_KEY,
                                  'structures.csv',
                                  parent_key='BASEDIR',
                                  typename='file')
                                  
        manifest_builder.add_path(self.STRUCTURE_TREE_KEY,
                                  'structures.json',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.STRUCTURE_UNIONIZES_KEY,
                                  'experiment_%d/structure_unionizes.csv',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.CCF_VERSION_KEY,
                                  '%s',
                                  parent_key='BASEDIR',
                                  typename='dir')

        manifest_builder.add_path(self.ANNOTATION_KEY,
                                  'annotation_%d.nrrd',
                                  parent_key=self.CCF_VERSION_KEY,
                                  typename='file')

        manifest_builder.add_path(self.TEMPLATE_KEY,
                                  'average_template_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.INJECTION_DENSITY_KEY,
                                  'experiment_%d/injection_density_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.INJECTION_FRACTION_KEY,
                                  'experiment_%d/injection_fraction_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.DATA_MASK_KEY,
                                  'experiment_%d/data_mask_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.PROJECTION_DENSITY_KEY,
                                  'experiment_%d/projection_density_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path(self.STRUCTURE_MASK_KEY,
                                  'structure_masks/resolution_%d/structure_%d.nrrd',
                                  parent_key=self.CCF_VERSION_KEY,
                                  typename='file')

        manifest_builder.write_json_file(file_name)
       
 
    @staticmethod
    def validate_structure_id(structure_id):

        try:
            structure_id = int(structure_id)
        except ValueError as e:
            raise ValueError("Invalid structure_id (%s): could not convert to integer." % str(structure_id))

        return structure_id


    @staticmethod
    def validate_structure_ids(structure_ids):

        for ii, sid in enumerate(structure_ids):
            structure_ids[ii] = MouseConnectivityCache.validate_structure_id(sid)

        return structure_ids
