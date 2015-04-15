# Copyright 2014 Allen Institute for Brain Science
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