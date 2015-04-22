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


class Microarray(Api):
    def __init__(self, base_uri=None):
        super(Microarray, self).__init__(base_uri)
    
    
    def build_rma_url_microarray_data_set_well_known_files(self, structure_acronyms=['DG'], set_name='HumanMA', donors=['H0351.2001']):
        '''Build a query relating samples to well knownfiles.
         
        :parameter structure_acronyms: the structure acronyms from the data set anatomy ontology
        :type structure_acronyms: list of strings
        :parameter set_name: data set name
        :type set_name: string
        :parameter donors: list of donor names
        :type donors: list of strings
        :returns: rma query url for the current rma entpoint
        :rtype: string
        '''
        return ''.join([self.rma_endpoint, 
                        '/query.json?q=',
                        'model::Sample,',
                        'rma::criteria,',
                        ("microarray_data_set(products[abbreviation$eq'%s']," % set_name),
                        ("specimen(donor[name$in'%s']))," % ','.join(donors)),
                        ("structure[acronym$in'%s']," % ','.join(structure_acronyms)),
                        'rma::include,',
                        'microarray_slides(well_known_files)'])
    
    
    def read_json_sample_microarray_slides_well_known_file_id(self, json_parsed_data):
        '''Get the list of well_known_file ids from a response body containing nested sample,microarray_slides,well_known_files.
        
        :parameter json_parsed_data: the json response from the Allen Institute Api RMA.
        :type json_parsed_data: hash
        :returns: a list of well_known_file ids
        :rtype: list of strings
        '''
        need_to_convert_to_dict = []
        
        if 'msg' in json_parsed_data:
            for slide in json_parsed_data['msg']:
                if 'microarray_slides' in slide:
                    for microarray_slide in slide['microarray_slides']:
                        if 'well_known_files' in microarray_slide:
                            for well_known_file in microarray_slide['well_known_files']:
                                if 'id' in well_known_file:
                                    need_to_convert_to_dict.append(str(well_known_file['id']))
        
        return need_to_convert_to_dict
    
    
    def get_sample_well_known_file_ids(self, structure_names=['DG']):
        '''Query the current RMA endpoint with a list of structure names
        to get the corresponding well known file ids.
        
        Returns
        -------
        list
            A list of well known file id strings.
        '''
        rma_builder_fn = self.build_rma_url_microarray_data_set_well_known_files
        json_traversal_fn = self.read_json_sample_microarray_slides_well_known_file_id
        
        return self.do_rma_query(rma_builder_fn, json_traversal_fn, structure_names) 