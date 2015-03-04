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

from pkg_resources import resource_filename
import os
import subprocess as sp
import logging
from allen_wrench.model.biophys_sim.config import Config


def choose_bps_command(command='bps_simple', conf_file=None):
    log = logging.getLogger('allen_wrench.model.biophys_sim.bps_command')

    log.info("bps command: %s" % (command))
    
    if conf_file:
        conf_file = os.path.abspath(conf_file)
    
    if command == 'help':
        print Config().argparser.parse_args(['--help'])
    elif command == 'nrnivmodl':
        sp.call(['nrnivmodl', 'modfiles']) # TODO: alternate location in manifest?
    elif command == 'run_simple':
        app_config = Config(unpack_lobs=['positions',
                                         'connections',
                                         'external_inputs'])
        description = app_config.load(conf_file)
        sys.path.insert(1, description.manifest.get_path('CODE_DIR'))
        (module_name, function_name) = description.data['runs'][0]['main'].split('#')
        run_module(description, module_name, function_name)
    elif command == 'run_model':
        from allen_wrench.config.app.mpi_job import MpiJob
        
        app_config = Config(unpack_lobs=['positions',
                                         'connections',
                                         'external_inputs'])
        description = app_config.load(conf_file)
        run_params = description.data['runs'][0]

        if 'num_mpi_processes' in run_params:
            num_mpi_processes = run_params['num_mpi_processes']
        else:
            num_mpi_processes = 4
        
        log.info('num mpi processes: %d' % 
                 (num_mpi_processes))
        
        start_module = resource_filename('allen_wrench.model.biophys_sim.bps_command',
                                         'bps_command.py')
        log.info("START_MODULE: %s" % (start_module))
        my_env = os.environ.copy()
        log.info("my env: %s" % (my_env))
        my_env['CONF_FILE'] = conf_file
        job = MpiJob(np=num_mpi_processes,
                     python='nrniv',
                     python_args=['-nobanner', '-mpi'],
                     module=start_module,
                     env=my_env)
        job.run()
    elif command == 'cluster_run_model':
        from allen_wrench.config.app.mpi_job import MpiJob
        
        app_config = Config(unpack_lobs=['positions',
                                         'connections',
                                         'external_inputs'])
        description = app_config.load(conf_file)
        run_params = description.data['runs'][0]

        pbs_args = description.data['cluster'][0]
        num_mpi_processes = 1
        
        if 'nodes' in pbs_args and 'ppn' in pbs_args:
            num_mpi_processes = int(pbs_args['nodes']) * int(pbs_args['ppn'])
        elif 'ncpus' in pbs_args:
            num_mpi_processes = pbs_args['ncpus']
                    
        log.info('num mpi processes: %d' % 
                 (num_mpi_processes))
        
        start_module = resource_filename('allen_wrench.model.biophys_sim.bps_command',
                                         'bps_command.py')
        log.info("START_MODULE: %s" % (start_module))
        my_env = os.environ.copy()
        log.info("my env: %s" % (my_env))
        my_env['CONF_FILE'] = conf_file
        job = MpiJob(np=num_mpi_processes,
                     python='nrniv',
                     python_args=['-nobanner', '-mpi'],
                     module=start_module,
                     env=my_env)
        job.run()
    elif command == 'run_model_cluster' or command == 'qsub_script':
        from allen_wrench.config.app.pbs_job import PbsJob
        
        app_config = Config(unpack_lobs=['positions',
                                         'connections',
                                         'external_inputs'])
        description = app_config.load(conf_file)
        pbs_args = description.data['cluster'][0]
        description.manifest.resolve_paths(pbs_args)
        pbs_args['script'] = '''\
export PATH=${PATH_PREFIX}:$PATH
#env
bps cluster_run_model
'''
        pbs_args['environment']['CONF_FILE'] = conf_file
        
        job = PbsJob(**pbs_args)
        
        if command == 'run_model_cluster':
            jobid = job.run()
            log.info("job id: %s" % (jobid))
        else:
            script = job.generate_script()
            print script
    elif command == 'nrnivmodl_cluster':
        app_config = Config(unpack_lobs=[])
        description = app_config.load(conf_file)

        modfiles_dir = resource_filename('biophys_sim', 'modfiles')
        pbs_args = description.data['cluster'][0]
        description.manifest.resolve_paths(pbs_args)
        pbs_args['script'] = 'cp -R %s .; nrnivmodl modfiles' % (modfiles_dir)
        pbs_args['nodes'] = 1
        pbs_args['ppn'] = 1
        job = PbsJob(**pbs_args)
    
        jobid = job.run()
        log.info("job id: %s" % (jobid))
    else:
        raise Exception("unknown command %s" %(command))


def run_module(description, module_name, function_name):
    m = __import__(module_name, fromlist=[function_name])
    func = getattr(m, function_name)
    
    func(description)


# this module is designed to be called from the bps script,
# which may use nrniv which does not pass in command line arguments
# So the configuration file path must be set in an environment variable.
if __name__ == '__main__':
    import sys
    conf_file = None
    argv = sys.argv

    if len(argv) > 1:
        if argv[0] == 'nrniv':
            command = 'run_simple'
        else:
            command = argv[1]
    else:        
        command = 'run_simple'
    
    if len(argv) > 2 and argv[-1].endswith('.conf'):
        conf_file = argv[-1]
    else:    
        try:
            conf_file = os.environ['CONF_FILE']
        except:
            pass
    
    choose_bps_command(command, conf_file)