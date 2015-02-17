# Copyright 2014 Allen Institute for Brain Science
# Licensed under the Allen Institute Terms of Use (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.alleninstitute.org/Media/policies/terms_of_use_content.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from allen_wrench.config.model.lob_parser import LobParser
import h5py
from scipy.sparse import csr_matrix
from numpy import array
import logging


class Hdf5LobParser(LobParser):
    def __init__(self):
        self.log = logging.getLogger(__name__)
    
    
    def read(self, file_path, *args, **kwargs):
        group_name = kwargs.get('group_name', 'data')

        table_type = None
        table_name = None
        table_data = None

        try:
            with h5py.File(file_path, 'r') as f:
                tables_group = f[group_name]
                
                for table_group in tables_group.values():
                    table_name = table_group['name'][()]
                    table_type = table_group['type'][()] # expected to be 'ndarray'                    

                    if table_type == 'ndarray':
                        table_data = array(table_group['data'])
                    elif table_type == 'csr_matrix':
                        csr_data = array(table_group['csr']['data'])
                        csr_indices = array(table_group['csr']['indices'])
                        csr_indptr = array(table_group['csr']['indptr'])
                        csr_shape = (table_group['csr']['shape'][0],
                                     table_group['csr']['shape'][1])
                        table_data = csr_matrix((csr_data,
                                                 csr_indices,
                                                 csr_indptr),
                                                shape=csr_shape)

                    else:
                        raise Exception("Unknown large object type: %s" % (table_type))
        except Exception:
            self.log.error("Couldn't read ABAPy HDF5 configuration: %s" % file_path)
            raise
    
        return table_data

    
    def write(self, file_path, data, *args, **kwargs):
        group_name = kwargs.get('group_name', 'data')
        table_type = kwargs.get('type', 'ndarray')
        table_name = kwargs.get('name', 'data')
        
        try:
            with h5py.File(file_path, 'w') as f:
                tables_group = f.create_group(group_name)
#                for table_name, table_data in simulation_configuration.data['tables'].items():                       
                table_group = tables_group.create_group(table_name)
                table_group['name'] = table_name
                table_group['type'] = table_type
                if table_type == 'csr_matrix':
                    csr = table_group.create_group('csr')
                    csr['data'] = data.data.tolist()
                    csr['indices'] = data.indices.tolist()
                    csr['indptr'] = data.indptr.tolist()
                    csr['shape'] = data.shape
                else:
                    table_group['data'] = data
        except Exception:
            self.log.warn("Couldn't write ABAPy HDF5 configuration: %s" % file_path)
            raise    
