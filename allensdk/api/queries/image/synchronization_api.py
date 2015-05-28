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

class SynchronizationApi(Api):
    def __init__(self, base_uri=None):
        super(SynchronizationApi, self).__init__(base_uri)
    
    
    def build_image_to_atlas_query(self,
                                   section_image_id,
                                   x, y,
                                   atlas_id,
                                   fmt='json'):
        '''
        
        Parameters
        ----------
        section_image_id : integer
            primary key
        x : float
            coordinate
        y : float
            coordinate
        atlas_id : int
            primary key
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.image_to_atlas_endpoint,
                       '/',
                       str(section_image_id),
                       '.',
                       fmt,
                       '?x=%f&y=%f' % (x, y),
                       '&atlas_id=',
                       str(atlas_id)])
        
        return url
    
    
    def build_image_to_image_2d_query(self,
                                     section_image_id,
                                     x, y,
                                     section_image_ids,
                                     fmt='json'):
        '''
        
        Parameters
        ----------
        section_image_id : integer
            what to retrieve
        x : float
            coordinate
        y : float
            coordinate
        section_image_ids : list of ints
            primary keys
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.image_to_image_2d_endpoint,
                       '/',
                       str(section_image_id),
                       '.',
                       fmt,
                       '?x=%f&y=%f' % (x, y),
                       '&section_image_ids=',
                       ','.join(str(i) for i in section_image_ids)])
        
        return url
    
    
    def build_reference_to_image_query(self,
                                       reference_space_id,
                                       x, y, z,
                                       section_data_set_ids,
                                       fmt='json'):
        '''
        
        Parameters
        ----------
        reference_space_id : integer
            primary key
        x : float
            coordinate
        y : float
            coordinate
        z : float
            coordinate
        section_data_set_ids : list of ints
            primary keys
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.reference_to_image_endpoint,
                       '/',
                       str(reference_space_id),
                       '.',
                       fmt,
                       '?x=%f&y=%f&z=%f' % (x, y, z),
                       '&section_data_set_ids=',
                       ','.join(str(i) for i in section_data_set_ids)])
        
        return url
    
    
    def build_image_to_reference_query(self,
                                       section_image_id,
                                       x, y,
                                       fmt='json'):
        '''
        
        Parameters
        ----------
        section_image_id : integer
            primary key
        x : float
            coordinate
        y : float
            coordinate
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.image_to_reference_endpoint,
                       '/',
                       str(section_image_id),
                       '.',
                       fmt,
                       '?x=%f&y=%f' % (x, y)])
        
        return url
    
    
    def build_structure_to_image_query(self,
                                       section_data_set_id,
                                       structure_ids,
                                       fmt='json'):
        '''
        
        Parameters
        ----------
        section_data_set_id : integer
            primary key
        structure_ids : list of integers
            primary key
        fmt : string, optional
            json (default) or xml
        
        Returns
        -------
        url : string
            The constructed URL
        '''
        url = ''.join([self.structure_to_image_endpoint,
                       '/',
                       str(section_data_set_id),
                       '.',
                       fmt,
                       '?structure_ids=',
                       ','.join([str(i) for i in structure_ids])])
        
        return url
    
    
    def read_data(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def get_image_to_atlas(self, section_image_id, x, y, atlas_id):
        '''Retrieve the structure graph data.'''
        sync_data = self.do_query(self.build_image_to_atlas_query,
                                  self.read_data,
                                  section_image_id,
                                  x, y,
                                  atlas_id)
        
        return sync_data
    
    
    def get_image_to_image_2d(self, section_image_id, x, y, section_image_ids):
        '''Retrieve the structure graph data.'''
        sync_data = self.do_query(self.build_image_to_image_2d_query,
                                  self.read_data,
                                  section_image_id,
                                  x, y,
                                  section_image_ids)
        
        return sync_data
    
    
    def get_reference_to_image(self, reference_space_id,
                               x, y, z,
                               section_data_set_ids):
        '''Retrieve the structure graph data.'''
        sync_data = self.do_query(self.build_reference_to_image_query,
                                  self.read_data,
                                  reference_space_id,
                                  x, y, z,
                                  section_data_set_ids)
        
        return sync_data
    
    
    def get_image_to_reference(self, 
                               section_image_id,
                               x, y):
        '''Retrieve.'''
        sync_data = self.do_query(self.build_image_to_reference_query,
                                  self.read_data,
                                  section_image_id,
                                  x, y)
        
        return sync_data
    
    
    def get_structure_to_image(self, 
                               section_data_set_id,
                               structure_ids):
        '''Retrieve.'''
        sync_data = self.do_query(self.build_structure_to_image_query,
                                  self.read_data,
                                  section_data_set_id,
                                  structure_ids)
        
        return sync_data



if __name__ == '__main__':
    # queries from http://help.brain-map.org/display/api/Image-to-Image+Synchronization
    import json
    from allensdk.api.queries.image.synchronization_api import SynchronizationApi
    
    a = SynchronizationApi()
    print(json.dumps(a.get_image_to_atlas(68173101, 6208, 2368, 1),
                     indent=2))
    print(json.dumps(a.get_image_to_image_2d(68173101,
                                             6208, 2368,
                                             [68173103, 68173105, 68173107]),
                     indent=2))
    print(json.dumps(a.get_reference_to_image(10,
                                              6085, 3670, 4883,
                                             [68545324, 67810540]),
                     indent=2))
    print(json.dumps(a.get_image_to_reference(68173101,
                                              6208, 2368),
                     indent=2))
    print(json.dumps(a.get_structure_to_image(68545324,
                                              [315,698,1089,703,477,803,512,549,1097,313,771,354]),
                     indent=2))
    

