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

import h5py
import numpy as np
from scipy.sparse import csr_matrix
import logging


class Hdf5Util(object):

    def __init__(self):
        self.log = logging.getLogger(__name__)

    def read(self, file_path):
        try:
            with h5py.File(file_path, 'r') as csr:
                return csr_matrix((csr['data'][...],
                                   csr['indices'][...],
                                   csr['indptr'][...]))
        except Exception:
            self.log.error(
                "Couldn't read AllenSDK HDF5 CSR configuration: %s" % file_path)
            raise

    def write(self, file_path, m):
        try:
            with h5py.File(file_path, 'w') as csr:
                csr.create_dataset('data', data=m.data, dtype=np.uint8)
                csr.create_dataset('indices', data=m.indices, dtype=np.uint32)
                csr.create_dataset('indptr', data=m.indptr, dtype=np.uint32)
        except Exception:
            self.log.warn(
                "Couldn't write AllenSDK HDF5 CSR configuration: %s" % file_path)
            raise
