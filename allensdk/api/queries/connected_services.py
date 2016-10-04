# Copyright 2015-2016 Allen Institute for Brain Science
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


class ConnectedServices(object):
    '''
    A class representing a schema of informatics web services.

    Notes
    -----
    See `Connected Services and Pipes <http://help.brain-map.org/display/api/Connected+Services+and+Pipes>`_
    for a human-readable list of services and parameters.

    The URL format is documented at
    `Service Pipelines <http://help.brain-map.org/display/api/Service+Pipelines>`_.

    Connected Services only include API services that are accessed
    via the RMA endpoint using an rma::services stage.
    '''
    ARRAY = 'array'
    STRING = 'string'
    INTEGER = 'integer'
    FLOAT = 'float'
    BOOLEAN = 'boolean'

    def __init__(self):
        pass

    def build_url(self, service_name, kwargs):
        '''Create a single stage RMA url from a service name and parameters.
        '''
        rma = RmaApi()
        fmt = kwargs.get('fmt', 'json')

        schema_entry = ConnectedServices._schema[service_name]

        params = []

        for parameter in schema_entry['parameters']:
            value = kwargs.get(parameter['name'], None)
            if value is not None:
                params.append((parameter['name'], value))

        service_stage = rma.service_stage(service_name,
                                          params)

        url = rma.build_query_url([service_stage], fmt)

        return url

    @classmethod
    @property
    def schema(cls):
        '''Dictionary of service names and parameters.

        Notes
        -----
        See `Connected Services and Pipes <http://help.brain-map.org/display/api/Connected+Services+and+Pipes>`_
        for a human-readable list of connected services and their parameters.
        '''
        return cls._schema

    _schema = {
        'dev_human_correlation': {
            'parameters': [
                {'name': 'set',
                 'optional': True,
                 'type': STRING,
                 'values': ['rna_seq_genes',
                            'rna_seq_exons',
                            'exon_microarray_genes'
                            'exon_microarray_exons']
                 },
                {'name': 'donors',
                 'optional': True,
                 'type': ARRAY
                 },
                {'name': 'structures',
                 'optional': False,
                 'type': ARRAY
                 },
                {'name': 'probes',
                 'optional': False,
                 'type': INTEGER
                 },
                {'name': 'sort_order',
                 'type': STRING,
                 'optional': True,
                 'values': ['asc', 'desc'],
                 'default': ['desc']
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'dev_human_differential': {
            'parameters': [
                {'name': 'set',
                 'type': STRING,
                 'values': ['rna_seq_genes',
                            'rna_seq_exons',
                            'exon_microarray_genes',
                            'exon_microarray_exons']
                 },
                {'name': 'donors1',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures1',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'donors2',
                 'type': ARRAY,
                 'optional': True,
                 'array_type': INTEGER
                 },
                {'name': 'structures2',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'sort_by',
                 'type': STRING,
                 'optional': True,
                 'values': ['p-value', 'fold-change'],
                 'default': 'p-value'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'dev_human_expression': {
            'parameters': [
                {'name': 'set',
                 'type': STRING,
                 'values': ['rna_seq_genes',
                            'rna_seq_exons',
                            'exon_microarray_genes',
                            'exon_microarray_exons']
                 },
                {'name': 'probes',
                 'type': INTEGER
                 },
                {'name': 'donors',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'dev_human_microarray_correlation': {
            'parameters': [
                {'name': 'donors',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING]
                 },
                {'name': 'probes',
                 'type': INTEGER
                 },
                {'name': 'sort_order',
                 'type': STRING,
                 'optional': True,
                 'values': ['asc', 'desc'],
                 'default': 'desc'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'dev_human_microarray_differential': {
            'parameters': [
                {'name': 'donors1',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures1',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'donors2',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures2',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'sort_by',
                 'type': STRING,
                 'optional': True,
                 'values': ['p-value', 'fold-change'],
                 'default': 'p-value'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'dev_human_microarray_expression': {
            'parameters': [
                {'name': 'probes',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 },
                {'name': 'donors',
                 'type': INTEGER,
                 'optional': True,
                 },
                {'name': 'structures',
                 'type': INTEGER,
                 'optional': True
                 }
            ]
        },
        'dev_mouse_agea': {
            'parameters': [
                {'name': 'seed_age',
                 'type': STRING
                 },
                {'name': 'map_age',
                 'type': STRING
                 },
                {'name': 'seed_point',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'seed_threshold',
                 'type': ARRAY,
                 'array_type': FLOAT
                 },
                {'name': 'map_threshold',
                 'type': ARRAY,
                 'array_type': FLOAT
                 },
                {'name': 'contrast_threshold',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'target_threshold',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'dev_mouse_correlation': {
            'parameters': [
                {'name': 'row',
                 'type': INTEGER
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'ages',
                 'type': ARRAY,
                 'array_type': STRING,
                 'optional': True
                 },
                {'name': 'sort_order',
                 'type': STRING,
                 'optional': True,
                 'values': ['asc', 'desc'],
                 'default': 'desc'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'gbm_correlation': {
            'parameters': [
                {'name': 'donors',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING]
                 },
                {'name': 'probes',
                 'type': INTEGER
                 },
                {'name': 'sort_order',
                 'type': STRING,
                 'optional': True,
                 'values': ['asc', 'desc'],
                 'default': 'desc'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'gbm_differential': {
            'parameters': [
                {'name': 'donors1',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures1',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'donors2',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures2',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'sort_by',
                 'type': STRING,
                 'optional': True,
                 'values': ['p-value', 'fold-change'],
                 'default': 'p-value'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'gbm_expression': {
            'parameters': [
                {'name': 'probes',
                 'type': INTEGER,
                 'array_type': INTEGER
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 },
                {'name': 'donors',
                 'type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures',
                 'type': INTEGER,
                 'optional': True
                 }
            ]
        },
        'gbm_ish_differential': {
            'parameters': [
                {'name': 'structures1',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'structures2',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'threshold1',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'threshold2',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'gbm_ish_expression': {
            'parameters': [
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'threshold',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'human_microarray_correlation': {
            'parameters': [
                {'name': 'donors',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING]
                 },
                {'name': 'probes',
                 'type': INTEGER
                 },
                {'name': 'sort_order',
                 'type': STRING,
                 'optional': True,
                 'values': ['asc', 'desc'],
                 'default': 'desc'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'human_microarray_differential': {
            'parameters': [
                {'name': 'donors1',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures1',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'donors2',
                 'type': ARRAY,
                 'optional': True,
                 'array_type': INTEGER
                 },
                {'name': 'structures2',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'sort_by',
                 'type': STRING,
                 'values': ['p-value', 'fold-change'],
                 'default': 'p-value'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'human_microarray_expression': {
            'parameters': [
                {'name': 'probes',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'donors',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'mouse_agea': {
            'parameters': [
                {'name': 'set',
                 'type': STRING
                 },
                {'name': 'seed_age',
                 'type': STRING
                 },
                {'name': 'map_age',
                 'type': STRING
                 },
                {'name': 'seed_point',
                 'type': ARRAY,
                 'array_type': FLOAT
                 },
                {'name': 'correlation_threshold1',
                 'type': FLOAT,
                 'optional': True,
                 },
                {'name': 'correlation_threshold2',
                 'type': FLOAT,
                 'optional': True
                 },
                {'name': 'threshold1',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'threshold2',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'mouse_correlation': {
            'parameters': [
                {'name': 'set',
                 'type': STRING,
                 'values': ['mouse', 'mouse_coronal']
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'row',
                 'type': INTEGER
                 },
                {'name': 'sort_order',
                 'type': STRING,
                 'optional': True,
                 'values': ['asc', 'desc'],
                 'default': 'desc'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'mouse_differential': {
            'parameters': [
                {'name': 'set',
                 'type': STRING,
                 'values': ['mouse', 'mouse_coronal']
                 },
                {'name': 'structures1',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'structures2',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'threshold1',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'threshold2',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'mouse_connectivity_correlation': {
            'parameters': [
                {'name': 'row',
                 'type': INTEGER
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'product_ids',
                 'type': ARRAY,
                 'array_type': [INTEGER],
                 'optional': True
                 },
                {'name': 'hemisphere',
                 'type': STRING,
                 'optional': True,
                 'values': ['right', 'left']
                 },
                {'name': 'transgenic_lines',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'injection_structures',
                 'type': 'Array',
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'primary_structure_only',
                 'type': BOOLEAN,
                 'optional': True,
                 },
                {'name': 'sort_order',
                 'type': STRING,
                 'optional': True,
                 'values': ['asc', 'desc'],
                 'default': 'desc'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'mouse_connectivity_injection_coordinate': {
            'parameters': [
                {'name': 'seed_point',
                 'type': ARRAY,
                 'array_type': FLOAT,
                 'optional': False,
                 },
                {'name': 'transgenic_lines',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'injection_structures',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'product_ids',
                 'type': ARRAY,
                 'array_type': [INTEGER],
                 'optional': True
                 },
                {'name': 'primary_structure_only',
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'mouse_connectivity_injection_structure': {
            'parameters': [
                {'name': 'injection_structures',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING]
                 },
                {'name': 'target_domain',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'injection_hemisphere',
                 'type': STRING,
                 'optional': True,
                 'values': ['right', 'left']
                 },
                {'name': 'target_hemisphere',
                 'type': STRING,
                 'optional': True,
                 'values': ['right', 'left']
                 },
                {'name': 'transgenic_lines',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'injection_domain',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'product_ids',
                 'type': ARRAY,
                 'array_type': [INTEGER],
                 'optional': True
                 },
                {'name': 'primary_structure_only',
                 'type': BOOLEAN,
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'mouse_connectivity_target_spatial': {
            'parameters': [
                {'name': 'seed_point',
                 'type': ARRAY,
                 'array_type': FLOAT,
                 },
                {'name': 'transgenic_lines',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'section_data_set',
                 'type': INTEGER,
                 'optional': True
                 },
                {'name': 'injection_structures',
                 'type': ARRAY,
                 'array_type': [INTEGER, STRING],
                 'optional': True
                 },
                {'name': 'product_ids',
                 'type': ARRAY,
                 'array_type': [INTEGER],
                 'optional': True
                 },
                {'name': 'primary_structure_only',
                 'type': BOOLEAN,
                 'optional': True
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'nhp_lmd_microarray_correlation': {
            'parameters': [
                {'name': 'donors',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 },
                {'name': 'probes',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'sort_order',
                 'type': STRING,
                 'optional': True,
                 'values': ['asc', 'desc'],
                 'default': 'desc'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'nhp_lmd_microarray_differential': {
            'parameters': [
                {'name': 'donors1',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures1',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'donors2',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures2',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'sort_by',
                 'type': STRING,
                 'optional': True,
                 'values': ['p-value', 'fold-change'],
                 'default': 'p-value'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'nhp_lmd_microarray_expression': {
            'parameters': [
                {'name': 'probes',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'nhp_macro_microarray_correlation': {
            'parameters': [
                {'name': 'donors',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'probes',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'sort_order',
                 'type': STRING,
                 'optional': True,
                 'values': ['asc', 'desc'],
                 'default': 'desc'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'nhp_macro_microarray_differential': {
            'parameters': [
                {'name': 'donors1',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures1',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'donors2',
                 'type': ARRAY,
                 'array_type': INTEGER,
                 'optional': True
                 },
                {'name': 'structures2',
                 'array_type': INTEGER,
                 'type': ARRAY
                 },
                {'name': 'sort_by',
                 'type': STRING,
                 'optional': True,
                 'values': ['p-value', 'fold-change'],
                 'default': 'p-value'
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'nhp_macro_microarray_expression': {
            'parameters': [
                {'name': 'probes',
                 'type': ARRAY,
                 'array_type': INTEGER
                 },
                {'name': 'start_row',
                 'type': INTEGER,
                 'optional': True,
                 'default': 0
                 },
                {'name': 'num_rows',
                 'type': INTEGER,
                 'optional': True,
                 'default': 2000
                 }
            ]
        },
        'text_search': {
            'parameters': [
                {'name': 'query_string',
                 'type': STRING
                 },
                {'name': 'k',
                 'type': STRING
                 }
            ]
        }
    }
