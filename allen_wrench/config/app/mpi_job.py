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

import os
import subprocess as sp
import logging


class MpiJob(object):
    log = logging.getLogger(__name__)
    mpi_command = 'mpiexec'
    
    def __init__(self, **kwargs):
        self.np = kwargs.get('np', '1')
        self.iface = kwargs.get('iface', None) # 'ib0'
        self.env = kwargs.get('env', os.environ.copy())
        self.module = kwargs.get('module', None)
        self.python = kwargs.get('python', 'nrniv')
        self.python_args = kwargs.get('python_args', ['-nobanner', '-mpi'])
    
    def run(self):
        subprocess_args = [MpiJob.mpi_command]
        
        subprocess_args.append('-np')
        subprocess_args.append(str(self.np))
        
        if self.iface != None:
            subprocess_args.append('-iface')
            subprocess_args.append(self.iface)
        
        subprocess_args.append(self.python)
        
        subprocess_args.extend(self.python_args)
        
        subprocess_args.append(self.module)
        
        sp.call(subprocess_args, env=self.env)