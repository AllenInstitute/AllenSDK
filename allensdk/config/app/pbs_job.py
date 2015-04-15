# Copyright 2014 Allen Institute for Brain Science
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

import subprocess as sp
import logging


class PbsJob(object):
    log = logging.getLogger(__name__)
    qsub_command = 'qsub'
    
    def __init__(self, **kwargs):
        self.job_name = kwargs.get('job_name', None)
        self.email = kwargs.get('email', None)
        self.email_options = kwargs.get('email_options', None)
        self.queue = kwargs.get('queue', None)
        self.vmem = kwargs.get('vmem', None)
        self.walltime = kwargs.get('walltime', '01:00:00')
        self.ncpus = kwargs.get('ncpus', None)
        self.ppn = kwargs.get('ppn', None)
        self.nodes = kwargs.get('nodes', None)
        self.job_dir = kwargs.get('job_dir', None)
        self.rerunable = kwargs.get('rerunable', False)
        self.output_dir = kwargs.get('output_dir', '$PBS_O_WORKDIR')
        self.out_file_name = kwargs.get('out_file', '$PBS_JOBID.out')
        self.err_file_name = kwargs.get('err_file', '$PBS_JOBID.err')
        self.script = kwargs.get('script', None)
        self.environment = kwargs.get('environment', {})
        
    
    def generate_script(self):
        script_lines = []
        
        script_lines.append('#!/bin/bash\n')
        script_lines.append('#PBS -q %s\n' % (self.queue))
        script_lines.append('#PBS -N %s\n' % (self.job_name))
        
        if self.email != None:
            script_lines.append('#PBS -M %s\n' % (self.email))
            
        if self.email_options != None:
            script_lines.append('#PBS -m %s\n' % (self.email_options))
        
        if self.rerunable == False:
            script_lines.append('#PBS -r n\n')
        
        if self.ncpus != None:
            ncpus_string = 'ncpus=%d' % self.ncpus
        else:
            ncpus_string = ''

        if self.ppn != None:
            ppn_string = 'ppn=%d' % self.ppn
        else:
            ppn_string = ''

        if self.nodes != None:
            nodes_string = 'nodes=%d:' % self.nodes
        else:
            nodes_string = ''
        
        tmp_str = '#PBS -l %s%s%s\n' % (ncpus_string, nodes_string, ppn_string)
        script_lines.append(tmp_str)
        
        vmem_walltime = []
        if self.vmem != None:
            vmem_walltime.append('vmem=%s' % (self.vmem))
            
        if self.walltime != None:
            vmem_walltime.append('walltime=%s' % (self.walltime))
            
        if len(vmem_walltime) > 0:
            tmp_str = '#PBS -l %s\n' % (','.join(vmem_walltime))
            script_lines.append(tmp_str)
        
        if self.job_dir != None:
            script_lines.append('#PBS -d %s\n' % (self.job_dir))
            
        script_lines.append('#PBS -o %s\n' % (self.out_file_name))
        script_lines.append('#PBS -e %s\n' % (self.err_file_name))
        
        env_list = []

        for variable, value in self.environment.items():
            env_list.append('%s=%s' % (variable, value))
        
        if len(env_list) > 0:
            script_lines.append('#PBS -v %s \n' % (','.join(env_list)))
        
        script_lines.append('%s\n' % (self.script))
        
        script_string = ''.join(script_lines)
        
        return script_string
    
    def run(self):
        sub_process = sp.Popen(PbsJob.qsub_command,
                               shell=True,
                               stdin=sp.PIPE,
                               stdout=sp.PIPE,
                               close_fds=True)
        sp_output = sub_process.stdout
        sp_input = sub_process.stdin
        
        if sp_input == None:
            raise Exception('could not start job')
        
        script_string = self.generate_script()
        
        PbsJob.log.info(script_string)
        
        sp_input.write(script_string)
        
        sp_input.close()
        
        return sp_output.read()