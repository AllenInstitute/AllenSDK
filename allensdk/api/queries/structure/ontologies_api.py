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

class OntologiesApi(Api):
    '''
    See: `Atlas Drawings and Ontologies <http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies>`_
    '''
    
    def __init__(self, base_uri=None):
        super(OntologiesApi, self).__init__(base_uri)
    
    
    def build_atlases_query(self, atlas_id=None, brief=True, fmt='json'):
        '''Build the URL to list Atlases available through the API
        with associated ontologies and structure graphs.
        
        Parameters
        ----------
        atlas_id : integer, optional
        brief : boolean, optional
            True (default) requests only name and id fields.
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        
        Notes
        -----
        This query is based on the
        `table of available Atlases <http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies>`_.
        See also: `Class: Atlas <http://api.brain-map.org/doc/Atlas.html>`_
        '''
        rma = RmaApi()
        
        associations = []
        
        if atlas_id != None:
            associations.append('[id$eq%d],' % (atlas_id))
        
        associations.extend(['structure_graph(ontology),',
                             'graphic_group_labels'])
        
        associations_string = ''.join(associations)
        
        if brief == True:
            only_fields = ['atlases.id',
                           'atlases.name',
                           'atlases.image_type',
                           'ontologies.id',
                           'ontologies.name',
                           'structure_graphs.id',
                           'structure_graphs.name',
                           'graphic_group_labels.id',
                           'graphic_group_labels.name']
            
            only_string = rma.quote_string(','.join(only_fields))
        
            atlas_model_stage = rma.model_stage('Atlas',
                                                include=[associations_string],
                                                criteria=[associations_string],
                                                only=[only_string])
        
        else:
            atlas_model_stage = rma.model_stage('Atlas',
                                                include=[associations_string],
                                                criteria=[associations_string])
        
        return rma.build_query_url(atlas_model_stage)
    
    
    def build_structure_query(self, graph_id):
        rma = RmaApi()
        
        # q = 'model::Structure[graph_id$eq1],rma::options[order$eqstructures.graph_order]&tabular=structures.id,structures.acronym,structures.graph_order,structures.color_hex_triplet,structures.structure_id_path,structures.name&start_row=0&num_rows=all'
        structure_model_stage = rma.model_stage('Structure',
                                                include=['[graph_id$eq%d]' % (graph_id)],
                                                order=['structures.graph_order'],
                                                tabular=['structures.id',
                                                         'structures.acronym',
                                                         'structures.graph_order',
                                                         'structures.color_hex_triplet',
                                                         'structures.structure_id_path',
                                                         'structures.name'],
                                                num_rows='all')
        
        return rma.build_query_url(structure_model_stage)

    
    def read_data(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def get_atlases_table(self, atlas_id=None, brief=True):
        '''List Atlases available through the API
        with associated ontologies and structure graphs.
        
        Parameters
        ----------
        atlas_id : integer, optional
        brief : boolean, optional
            True (default) requests only name and id fields.
        
        Returns
        -------
        data : dict
            The parsed json response from the API
        '''
        data = self.do_query(self.build_atlases_query,
                            self.read_data,
                            atlas_id,
                            brief)
        
        return data
    
    
    def get_ontology(self, structure_graph_id):
        '''Retrieve.'''
        data = self.do_query(self.build_query,
                                   self.read_data)
        
        return data
    
    
    def get_structures(self, structure_graph_id):
        data = self.do_query(self.build_structure_query,
                             self.read_data,
                             structure_graph_id)
        
        return data


if __name__ == '__main__':
    print(OntologiesApi().get_atlases_table(1))