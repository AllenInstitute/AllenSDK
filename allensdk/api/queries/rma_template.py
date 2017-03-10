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
from jinja2 import Template


class RmaTemplate(RmaApi):
    '''
    See: `Atlas Drawings and Ontologies
    <http://help.brain-map.org/display/api/Atlas+Drawings+and+Ontologies>`_
    '''

    def __init__(self, base_uri=None, query_manifest=None):
        super(RmaTemplate, self).__init__(base_uri)
        self.templates = query_manifest

    def to_filter_rhs(self, rhs):
        if type(rhs) == list:
            return ','.join(str(r) for r in rhs)

        return rhs

    def template_query(self, template_name, entry_name, **kwargs):
        cb = self.templates[template_name]
        templates = [e for e in cb if e['name'] == entry_name]

        if len(templates) > 0:
            template = templates[0]
        else:
            raise Exception('Entry %s not found.' % (entry_name))

        query_args = {'model': template['model']}

        if 'criteria' in template:
            criteria_template = Template(template['criteria'])

            if 'criteria_params' in template:
                criteria_params = {key: self.to_filter_rhs(kwargs.get(key))
                                   for key in template['criteria_params']
                                   if key in kwargs and kwargs.get(key) is not None}
            else:
                criteria_params = {}

            criteria_str = str(criteria_template.render(**criteria_params))
            if criteria_str:
                query_args['criteria'] = criteria_str

        if 'include' in template:
            include_template = Template(template['include'])

            if 'include_params' in template:
                include_params = {key: self.to_filter_rhs(kwargs.get(key))
                                  for key in template['include_params']
                                  if key in kwargs and kwargs.get(key) is not None}
            else:
                include_params = {}

            include_str = str(include_template.render(**include_params))
            if include_str:
                query_args['include'] = include_str

        if 'only' in kwargs:
            if kwargs.get('only') is not None:
                query_args['only'] = [self.quote_string(
                    ','.join(kwargs.get('only')))]
        elif 'only' in template:
            query_args['only'] = [
                self.quote_string(','.join(template['only']))]

        if 'except' in kwargs:
            if kwargs.get('except') is not None:
                query_args['except'] = [self.quote_string(
                    ','.join(kwargs.get('except')))]
        elif 'except' in template:
            query_args['except'] = template['except']

        if 'start_row' in kwargs:
            query_args['start_row'] = kwargs.get('start_row')
        elif 'start_row' in template:
            query_args['start_row'] = template['start_row']

        if 'num_rows' in kwargs:
            query_args['num_rows'] = kwargs.get('num_rows')
        elif 'num_rows' in template:
            query_args['num_rows'] = template['num_rows']

        if 'count' in kwargs:
            query_args['count'] = kwargs.get('count')
        elif 'count' in template:
            query_args['count'] = template['count']

        if 'order' in kwargs:
            query_args['order'] = kwargs.get('order')
        elif 'order' in template:
            query_args['order'] = template['order']

        query_args.update(kwargs)

        data = self.model_query(**query_args)

        return data
