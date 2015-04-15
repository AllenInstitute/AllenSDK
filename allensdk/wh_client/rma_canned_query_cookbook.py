# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

class RmaCannedQueryCookbook:
    def __init__(self, rma_endpoint):
        '''A class that generates canned RMA queries for common Warehouse tasks.
        This should be considered to be a collection of examples rather than a complete library.
        '''
        self.rma_endpoint = rma_endpoint

        
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
        
        :parameter json_parsed_data: the json response from the Allen Institute Warehouse RMA.
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


    def build_rma_biophysical_model_well_known_files(self, model_names=['DG']):
        '''Build a query relating biophysical models to well known files.
         
        :parameter model_names: the biophysical model names
        :type model_names: list of strings
        :returns: rma query url for the current rma entpoint
        :rtype: string
        '''
        return ''.join([self.rma_endpoint, 
                        '/query.json?q=',
                        'model::BiophysicalModel,',
                        'rma::criteria,',
                        ("[name$in'%s']," % "','".join(model_names)),
                        'rma::include,',
                        'well_known_files[id]'])
                        
    def read_json_biophysical_model_well_known_file_id(self, json_parsed_data):
        '''Get the list of well_known_file ids from a response body containing nested biophysical_models,well_known_files.
        
        :parameter json_parsed_data: the json response from the Allen Institute Warehouse RMA.
        :type json_parsed_data: hash
        :returns: a dict of biophysical_model name to well-known-file id mappings
        :rtype: dict
        '''
        model_to_file_id_dict = {}
        
        if 'msg' in json_parsed_data:
            for biophysical_model in json_parsed_data['msg']:
                        current_model_name = None
                        
                        if 'name' in biophysical_model:
                            current_model_name = biophysical_model['name']
                        if 'well_known_files' in biophysical_model:
                            for well_known_file in biophysical_model['well_known_files']:
                                if 'id' in well_known_file:
                                    model_to_file_id_dict[current_model_name] = str(well_known_file['id'])
        
        return model_to_file_id_dict
        
        
    def build_rma_url_biophysical_neuronal_model_run(self, neuronal_model_run_id, fmt='json'):
        include_associations = ''.join([
            'neuronal_model',
            '(neuronal_model_template,',
            'specimen(neuron_reconstructions(well_known_files),',
            'ephys_sweeps),',
            'well_known_files),',
            'well_known_files'])
        criteria_associations = ''.join([
            ("[id$eq%d]," % (neuronal_model_run_id)),
            include_associations])
        
        return ''.join([self.rma_endpoint, 
                        '/query.',
                        fmt,
                        '?q=',
                        'model::NeuronalModelRun,',
                        'rma::criteria,',
                        criteria_associations,
                        ',rma::include,',
                        include_associations])
