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

class AnnotatedSectionDataSetsApi(Api):
    def __init__(self, base_uri=None):
        super(AnnotatedSectionDataSetsApi, self).__init__(base_uri)
    
    def build_query(self,
                    structures,
                    intensity_values=None,
                    density_values=None,
                    pattern_values=None,
                    age_names=None,
                    fmt='json'):
        '''Build the URL.
        
        Parameters
        ----------
        structure_graph_id : dict of integers
            what to retrieve
        intensity_values : array of strings, optional
            intensity values, 'High','Low', 'Medium' (default)
        density_values : array of strings, optional
            density values, 'High', 'Low'
        pattern_values : array of strings, optional
            pattern values, 'Full'
        age_names : array of strings, options
            for example 'E11.5', '13.5'
        fmt : string, optional
            'json' or 'xml'
        
        Returns
        -------
        url : string
            The constructed URL
        
        See Also
        --------
        `http://help.brain-map.org/display/api/Searching+Annotated+SectionDataSets`
        '''
        params = ['structures=' + ','.join((str(s) for s in structures))]
        
        if intensity_values is not None and len(intensity_values) > 0:
            params.append('intensity_values=' +
                          ','.join(("'%s'" % (v) for v in intensity_values)))
        
        if density_values is not None and len(density_values) > 0:
            params.append('density_values=' +
                          ','.join(("'%s'" % (v) for v in density_values)))
        
        if pattern_values is not None and len(pattern_values) > 0:
            params.append('pattern_values=' +
                          ','.join(("'%s'" % (v) for v in pattern_values)))
        
        if age_names is not None and len(age_names) > 0:
            params.append('age_names=' +
                          ','.join(("'%s'" % (v) for v in age_names)))
        
        url_params = '?' + '&'.join(params)

        url = ''.join([self.annotated_section_data_sets_endpoint,
                       '.',
                       fmt,
                       url_params])
        
        return url
    
    
    def build_rma_query(self,
                        structures,
                        intensity_values=None,
                        density_values=None,
                        pattern_values=None,
                        age_names=None,
                        fmt='json'):
        '''Build the URL.
        
        Parameters
        ----------
        structure_graph_id : dict of integers
            what to retrieve
        intensity_values : array of strings, optional
            intensity values, 'High','Low', 'Medium' (default)
        density_values : array of strings, optional
            density values, 'High', 'Low'
        pattern_values : array of strings, optional
            pattern values, 'Full'
        age_names : array of strings, options
            for example 'E11.5', '13.5'
        fmt : string, optional
            'json' or 'xml'
        
        Returns
        -------
        url : string
            The constructed URL
        
        See Also
        --------
        `http://help.brain-map.org/display/api/Searching+Annotated+SectionDataSets`
        '''
        
        age_include_strings = ['age']
        
        if age_names is not None and len(age_names) > 0:
            age_include_strings.append('[name$in')
            age_include_strings.append(
                ','.join(("'%s'" % (a) for a in age_names)))
            age_include_strings.append(']')
        age_include = ''.join(age_include_strings)
        
        model_clause = 'model::SectionDataSet'
        
        criteria_strings = ['rma::criteria,',
                            'manual_annotations']
        
        if intensity_values is not None and len(intensity_values) > 0:
            criteria_strings.append('[intensity_call$in%s]' %
                (','.join(("'%s'" % (v) for v in intensity_values))))
        
        if density_values is not None and len(density_values) > 0:
            criteria_strings.append('[density_call$in%s]' %
                (','.join(("'%s'" % (v) for v in density_values))))
        
        if pattern_values is not None and len(pattern_values) > 0:
            criteria_strings.append('[pattern_call$in%s]' %
                (','.join(("'%s'" % (v) for v in pattern_values))))
        
        criteria_strings.append('(structure[id$in%s])' % 
            (','.join((str(s) for s in structures))))
        
        criteria_clause = ''.join(criteria_strings)
        
        include_clause = ''.join(['rma::include,',
                                  'specimen',
                                  '(donor(',
                                  age_include,
                                  ')),',
                                  'probes(gene),'
                                  'plane_of_section'])
        
        num_rows_clause = '[num_rows$eq50]'
        start_row_clause = '[start_row$eq0]'
        order_by_array = ['genes.acronym',
                          'ages.embryonic+desc',
                          'ages.days',
                          'data_sets.id']
        order_clause = '[order$eq%s]' % (
            ','.join(("'%s'" % (o) for o in order_by_array)))
        
        options_clause = ''.join(['rma::options',
                                  num_rows_clause,
                                  start_row_clause,
                                  order_clause])
        
        rma_query = ','.join([model_clause,
                              criteria_clause,
                              include_clause,
                              options_clause])
        
        url = ''.join([self.rma_endpoint,
                       '/query.',
                       fmt,
                       '?q=',
                       rma_query])
        
        return url
    
    
    def build_compound_query(self, queries, fmt='json'):
        '''Build the URL.
        
        Parameters
        ----------
        queries : array of dicts
            dicts with args like build_query
        fmt : string, optional
            'json' or 'xml'
        
        Returns
        -------
        url : string
            The constructed URL
        
        See Also
        --------
        `http://help.brain-map.org/display/api/Searching+Annotated+SectionDataSets`
        '''
        url_strings = ['?query=']
        
        for query in queries:
            url_strings.append('[')
            
            params = ['structures $in ' +
                      ','.join((str(s) for s in query['structures']))]
            
            for key in ['intensity_values', 'density_values', 'pattern_values', 'age_names']:
                if key in query and len(query[key]) > 0:
                    params.append('%s $in %s' %
                                  (key,
                                   ','.join(("'%s'" % (v) for v in query['intensity_values']))))
            
            url_strings.append(' : '.join(params))
            
            url_strings.append(']')
            
            if 'link' in query and query['link'] == 'or':
                url_strings.append(' or ')
            if 'link' in query and query['link'] == 'and':
                url_strings.append(' and ')
        
        url_params = ''.join(url_strings)
        
        url = ''.join([self.compound_annotated_section_data_sets_endpoint,
                       '.',
                       fmt,
                       url_params])
        
        return url
    
    
    def read_data(self, parsed_json):
        '''Return the list of cells from the parsed query.
        
        Parameters
        ----------
        parsed_json : dict
            A python structure corresponding to the JSON data returned from the API.
        '''
        return parsed_json['msg']
    
    
    def get_annotated_section_data_sets(self,
                                        structures,
                                        intensity_values=None,
                                        density_values=None,
                                        pattern_values=None,
                                        age_names=None):
        '''Retrieve the annotated section data sets data.'''
        data = self.do_query(self.build_query,
                             self.read_data,
                             structures,
                             intensity_values,
                             density_values,
                             pattern_values,
                             age_names)
        
        return data
    
    
    def get_annotated_section_data_sets_via_rma(self,
                                                structures,
                                                intensity_values=None,
                                                density_values=None,
                                                pattern_values=None,
                                                age_names=None):
        '''Retrieve the annotated section data sets data.'''
        data = self.do_query(self.build_rma_query,
                             self.read_data,
                             structures,
                             intensity_values,
                             density_values,
                             pattern_values,
                             age_names)
        
        return data
    
    
    def get_compound_annotated_section_data_sets(self,
                                                 queries):
        '''Retrieve the annotated section data sets data.'''
        data = self.do_query(self.build_compound_query,
                             self.read_data,
                             queries)
        
        return data
    
