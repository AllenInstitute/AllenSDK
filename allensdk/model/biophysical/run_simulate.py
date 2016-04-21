import allensdk.model.biophysical.runner as single_cell
import logging, os, sys, traceback, subprocess, logging.config as lc
import shutil
from allensdk.model.biophys_sim.config import Config
from pkg_resources import resource_filename #@UnresolvedImport


class RunSimulate(object):
    _log = logging.getLogger('allensdk.model.biophysical.run_simulate')
    
    def __init__(self,
                 input_json, 
                 output_json):
        self.input_json = input_json
        self.output_json = output_json
        self.app_config = None
        self.manifest = None
        
    
    def load_manifest(self):
        self.app_config = Config().load(self.input_json)
        self.manifest = self.app_config.manifest
        #combined_manifest = 'manifest.json'
        fix_sections = ['passive', 'axon_morph,', 'conditions', 'fitting']
        #self.description = self.app_config.load(combined_manifest)
        self.app_config.fix_unary_sections(fix_sections)
    
    
    def generate_manifest_rma(self,
                              neuronal_model_run_id,
                              manifest_path):
        '''
        Note
        ----
        Other necessary files are also written.
        '''
        import json
        from allensdk.api.api import Api
        from allensdk.internal.api.queries.biophysical_module_api import BiophysicalModuleApi
        from allensdk.internal.api.queries.biophysical_module_reader import BiophysicalModuleReader        
        Api.default_api_url = 'http://axon:3000'
        
        bma = BiophysicalModuleApi()
        data = bma.get_neuronal_model_runs(neuronal_model_run_id)
        
        lr = BiophysicalModuleReader()
        lr.read_lims_message(data, 'lims_message.json')
        
        with open('lims_message.json', 'w') as f:
            f.write(json.dumps(data[0], sort_keys=True, indent=2))
        
        lr.to_manifest(manifest_path) 


    def generate_manifest_lims(self,
                               lims_data_path,
                               manifest_path):
        '''
        Note
        ----
        Other necessary files are also written.
        '''
        from allensdk.internal.api.queries.biophysical_module_reader import BiophysicalModuleReader
            
        self.lims_json = lims_data_path
                       
        lr = BiophysicalModuleReader()
        lr.read_lims_file(self.lims_json)
        
        lr.to_manifest(manifest_path) 


    def copy_local(self):
        import allensdk.model.biophysical.run_simulate
        
        self.load_manifest()
        
        modfile_dir = self.manifest.get_path('MODFILE_DIR')
        
        if not os.path.exists(modfile_dir):
            os.mkdir(modfile_dir)
                    
        workdir = self.manifest.get_path('WORKDIR')        
        
        if not os.path.exists(workdir):
            os.mkdir(workdir)

        modfiles = [self.manifest.get_path(key) for key,info
                    in self.manifest.path_info.items()
                    if 'format' in info and info['format'] == 'MODFILE']
        
        for from_file in modfiles:             
            RunSimulate._log.debug("copying %s to %s" % (from_file, modfile_dir))
            shutil.copy(from_file, modfile_dir)
        
        shutil.copy(self.manifest.get_path('fit_parameters'),
                    workdir)
        
        shutil.copyfile(self.manifest.get_path('stimulus_path'),
                        self.manifest.get_path('output_path'))
        
        shutil.copy(resource_filename(allensdk.model.biophysical.run_simulate.__name__,
                                      'cell.hoc'),
                    os.curdir)
    
        
    def nrnivmodl(self):
        RunSimulate._log.debug("nrnivmodl")
        
        subprocess.call(['nrnivmodl', './modfiles'])    
    
    def simulate(self):
        from allensdk.internal.api.queries.biophysical_module_reader import BiophysicalModuleReader
        
        self.load_manifest()
                
        try:
            stimulus_path = self.manifest.get_path('stimulus_path')
            RunSimulate._log.info("stimulus path: %s" % (stimulus_path))
        except:
            raise Exception('Could not read input stimulus path from input config.')
        
        try:
            out_path = self.manifest.get_path('output_path')
            RunSimulate._log.info("result NWB file: %s" % (out_path))
        except:
            raise Exception('Could not read output path from input config.')
        
        try:
            morphology_path = self.manifest.get_path('MORPHOLOGY')
            RunSimulate._log.info("morphology path: %s" % (morphology_path))
        except:
            raise Exception('Could not read morphology path from input config.')
        
        try:
            sweeps = self.app_config.sweep_numbers()
            RunSimulate._log.info("sweeps from input: %s" % (','.join(str(s) for s in sweeps)))
        except:
            RunSimulate._log.warn('Could not read sweep numbers from input config.')
        
        
        single_cell.run(self.app_config)
        
        lims_upload_config = BiophysicalModuleReader()
        lims_upload_config.read_json(self.manifest.get_path('neuronal_model_run_data'))
        lims_upload_config.add_well_known_file(BiophysicalModuleReader.STIMULUS_CONTENT_TYPE,
                                               out_path)
        lims_upload_config.set_workflow_state('passed')
        lims_upload_config.write_file(self.output_json)
    
    
def main(command, lims_strategy_json, lims_response_json):
    ''' Entry point for module.
        :param command: select behavior, nrnivmodl or simulate
        :type command: string    
        :param lims_strategy_json: path to json file output from lims.
        :type lims_strategy_json: string
        :param lims_response_json: path to json file returned to lims.
        :type lims_response_json: string
    '''
    rs = RunSimulate(lims_strategy_json,
                     lims_response_json)

    RunSimulate._log.debug("command: %s" % (command))
    RunSimulate._log.debug("lims strategy json: %s" % (lims_strategy_json))
    RunSimulate._log.debug("lims upload json: %s" % (lims_response_json))
        
    log_config = resource_filename('allensdk.model.biophysical.run_simulate',
                                    'logging.conf')
    lc.fileConfig(log_config)
    os.environ['LOG_CFG'] = log_config    
    
    if 'nrnivmodl' == command:
        rs.nrnivmodl()
    elif 'copy_local' == command:
        rs.copy_local()        
    elif 'generate_manifest_rma' == command:
        rs.generate_manifest_rma(input_json, output_json)
    elif 'generate_manifest_lims' == command:
        rs.generate_manifest_lims(input_json, output_json)        
    else:
        rs.simulate()


if __name__ == '__main__':
    command, input_json, output_json = sys.argv[-3:]
    
    try:
        main(command, input_json, output_json)
        RunSimulate._log.debug("success")
    except Exception as e:
        RunSimulate._log.error(traceback.format_exc())   
        exit(1)
