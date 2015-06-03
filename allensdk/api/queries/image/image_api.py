# Copyright 2015 Allen Institute for Brain Science
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

from allensdk.api.api import Api
from allensdk.api.queries.rma.rma_api import RmaApi

class ImageApi(Api):
    '''
    See: `Atlas Drawings and Ontologies <http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies>`_
    '''
    def __init__(self, base_uri=None):
        super(ImageApi, self).__init__(base_uri)
    
    
    def build_atlas_image_query(self, atlas_id, image_type_name=None, fmt='json'):
        '''Build the URL.
        
        Parameters
        ----------
        atlas_id : integer, optional
            request a certain record.
        image_type_name : string, optional
            if not present, the query will get it from the atlas id.
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        
        Notes
        -----
        See `Downloading Atlas Images and Graphics <http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies#AtlasDrawingsandOntologies-DownloadingAtlasImagesAndGraphics>`_
        for additional documentation.
        The atlas id can be found with :py:meth:`allensdk.api.queries.structure.ontologies_api.OntologiesApi.build_atlases_query`
        '''
        rma = RmaApi()
        
        stages = []
        
        if image_type_name == None:
            atlas_stage = rma.model_stage('Atlas',
                                          criteria='[id$eq%d]' % (atlas_id),
                                          only=['image_type'])
            stages.append(atlas_stage)
            
            atlas_name_pipe_stage = rma.pipe_stage('list',
                                                   parameters=[('type_name',
                                                               rma.IS,
                                                               rma.quote_string('image_type'))])
            stages.append(atlas_name_pipe_stage)
            
            image_type_name = '$type_name'
        else:
            image_type_name = rma.quote_string(image_type_name)
        
        criteria_list = ['[annotated$eqtrue],',
                         'atlas_data_set(atlases[id$eq%d]),' % (atlas_id),
                         "alternate_images[image_type$eq%s]" % (image_type_name)]
        
        atlas_image_model_stage = rma.model_stage('AtlasImage',
                                                  criteria=criteria_list,
                                                  order=['sub_images.section_number'],
                                                  num_rows='all')
        
        stages.append(atlas_image_model_stage)
        
        return rma.build_query_url(stages)
    
    
    def read_data(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def get_ontology(self, structure_graph_id):
        '''Retrieve.'''
        data = self.do_query(self.build_query,
                                   self.read_data)
        
        return data

if __name__ == '__main__':
    #print(ImageApi().build_atlas_image_query(1))
    print(ImageApi().build_atlas_image_query(138322605))
    # This version doesn't build a pipeline, since the atlas name is explicit
    #print(ImageApi().build_atlas_image_query(1, 'Atlas - Adult Mouse'))

