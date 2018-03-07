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

from allensdk.core import sitk_utilities
from allensdk.api.cache import Cache, cacheable

from .reference_space_api import ReferenceSpaceApi
from .grid_data_api import GridDataApi
from .rma_pager import pageable



class MouseAtlasApi(ReferenceSpaceApi, GridDataApi):

    MOUSE_ATLAS_PRODUCTS = (1,)
    DEVMOUSE_ATLAS_PRODUCTS = (3,)
    MOUSE_ORGANISM = (1,)

    @cacheable()
    @pageable(num_rows=2000, total_rows='all')
    def get_section_data_sets(self, gene_ids=None, product_ids=None, **kwargs):
        '''
        '''
        
        if product_ids is None:
            list(self.MOUSE_ATLAS_PRODUCTS)
        criteria = 'products[id$in{}]'.format(','.join(map(str, product_ids)))

        if gene_ids is not None:
            gene_ids = cs_string(gene_ids)
            criteria += ',genes[id$in{}]'.format(','.join(map(str, gene_ids)))

        return self.model_query(model='SectionDataSet', 
                                criteria=criteria,
                                include='genes',
                                only=['id', 'genes.id', 'genes.name', 'reference_space_id'],
                                **kwargs)

    @cacheable()
    @pageable(num_rows=2000, total_rows='all')
    def get_genes(self, organism_ids=None, chromosome_ids=None, **kwargs):
        '''
        '''

        if organism_ids is None:
            organism_ids = list(self.MOUSE_ORGANISM)
        criteria = '[organism_id$in{}]'.format(','.join(map(str, organism_ids)))

        if chromosome_ids is not None:
            criteria += ',[chromosome_id$in{}]'.format(','.join(map(str, chromosome_ids)))
        
        return self.model_query(model='Gene', 
                                criteria=criteria,
                                include='chromosome',
                                only=['id', 'name', 'organism_id', 'acronym','chromosome.id', 'chromosome.name'],
                                **kwargs)

    @cacheable(strategy='create', 
               reader = sitk_utilities.read_ndarray_with_sitk, 
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_expression_density(self, path, experiment_id):
        self.download_gene_expression_grid_data(
            experiment_id, GridDataApi.DENSITY, path)


    @cacheable(strategy='create', 
               reader = sitk_utilities.read_ndarray_with_sitk, 
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_expression_energy(self, path, experiment_id):
        self.download_gene_expression_grid_data(
            experiment_id, GridDataApi.ENERGY, path)


    @cacheable(strategy='create', 
               reader = sitk_utilities.read_ndarray_with_sitk, 
               pathfinder=Cache.pathfinder(file_name_position=1,
                                           path_keyword='path'))
    def download_expression_intensity(self, path, experiment_id):
        self.download_gene_expression_grid_data(
            experiment_id, GridDataApi.INTENSITY, path)