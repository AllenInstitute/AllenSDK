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
from pandas import DataFrame, read_hdf
import logging


class PandasLobParser(LobParser):
    def __init__(self):
        self.log = logging.getLogger(__name__)
    
    
    def read(self, file_path, *args, **kwargs):
        group_name = kwargs.get('group_name', 'data')
        
        data_frame = read_hdf(file_path, group_name)
        
        return data_frame
    
    
    def write(self, file_path, data, *args, **kwargs):
        '''
        :param data: a list of dicts
        '''
        group_name = kwargs.get('group_name', 'data')
        mode = kwargs.get('mode', 'w')
        
        wrapped = dict([(i, v) for i, v in enumerate(data)])
        data_frame = DataFrame.from_dict(wrapped, orient='index')
        
        data_frame.to_hdf(file_path, group_name, mode=mode)
