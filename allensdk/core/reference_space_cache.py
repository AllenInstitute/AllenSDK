# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2015-2017. Allen Institute. All rights reserved.
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


from allensdk.config.manifest_builder import ManifestBuilder
from allensdk.api.cache import Cache
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi


class ReferenceSpaceCache(cache):


        
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
            post=lambda x: StructureTree(StructureTree.clean_structures(x)), 
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
        This method downloads structure masks from the Allen Institute. To make your own locally, see 
        ReferenceSpace.many_structure_masks.
        
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

        return self.api.download_structure_mask(structure_id, 
                                                self.ccf_version,
                                                self.resolution,
                                                file_name, 
                                                strategy='lazy')
