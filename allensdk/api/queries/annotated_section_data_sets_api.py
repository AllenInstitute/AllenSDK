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

from .rma_api import RmaApi
from ..cache import cacheable


class AnnotatedSectionDataSetsApi(RmaApi):
    '''See:
    `Searching Annotated SectionDataSets <http://help.brain-map.org/display/api/Searching+Annotated+SectionDataSets>`_
    '''

    def __init__(self, base_uri=None):
        super(AnnotatedSectionDataSetsApi, self).__init__(base_uri)

    def get_annotated_section_data_sets(self,
                                        structures,
                                        intensity_values=None,
                                        density_values=None,
                                        pattern_values=None,
                                        age_names=None):
        '''For a list of target structures, find the SectionDataSet
        that matches the parameters for intensity_values, density_values, pattern_values, and Age.

        Parameters
        ----------
        structure_graph_id : dict of integers
            what to retrieve
        intensity_values : array of strings, optional
            'High','Low', 'Medium' (default)
        density_values : array of strings, optional
            'High', 'Low'
        pattern_values : array of strings, optional
            'Full'
        age_names : array of strings, options
            for example 'E11.5', '13.5'

        Returns
        -------
        data : dict
            The parsed JSON repsonse message.

        Notes
        -----
        This method uses the non-RMA Annotated SectionDataSet endpoint.
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
                       '.json',
                       url_params])

        return self.json_msg_query(url)

    @cacheable()
    def get_annotated_section_data_sets_via_rma(self,
                                                structures,
                                                intensity_values=None,
                                                density_values=None,
                                                pattern_values=None,
                                                age_names=None):
        '''For a list of target structures, find the SectionDataSet
        that matches the parameters for intensity_values, density_values, pattern_values, and Age.

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

        Returns
        -------
        data : dict
            The parsed JSON response message.

        Notes
        -----
        This method uses the RMA endpoint to search annotated SectionDataSet data.
        '''
        age_include_strings = ['age']

        if age_names is not None and len(age_names) > 0:
            age_include_strings.append('[name$in')
            age_include_strings.append(
                ','.join(("'%s'" % (a) for a in age_names)))
            age_include_strings.append(']')
        age_include = ''.join(age_include_strings)

        criteria_strings = ['manual_annotations']

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

        include_clause = ''.join(['specimen',
                                  '(donor(',
                                  age_include,
                                  ')),',
                                  'probes(gene),'
                                  'plane_of_section'])

        order_by_array = ['genes.acronym',
                          'ages.embryonic+desc',
                          'ages.days',
                          'data_sets.id']

        data = self.model_query('SectionDataSet',
                                criteria=criteria_clause,
                                include=include_clause,
                                start_row=0,
                                num_rows=50,
                                order=order_by_array)

        return data

    def get_compound_annotated_section_data_sets(self,
                                                 queries,
                                                 fmt='json'):
        '''Find the SectionDataSet that matches several annotated_section_data_sets queries
        linked together with a Boolean 'and' or 'or'.

        Parameters
        ----------
        queries : array of dicts
            dicts with args like build_query
        fmt : string, optional
            'json' or 'xml'

        Returns
        -------
        data : dict
            The parsed JSON repsonse message.
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

        return self.json_msg_query(url)
