import os, shutil, subprocess, logging, logging.config as lc


import allensdk.core.json_utilities as ju
from allensdk.core.nwb_data_set import NwbDataSet
from pkg_resources import resource_filename #@UnresolvedImport
import allensdk.model.biophysical as hoc_location
from allensdk.api.queries.lims.optimize_config_reader import OptimizeConfigReader
from allensdk.model.biophys_sim.config import Config
from allensdk.model.biophysical.make_deap_fit_json import Report


class RunOptimize(object):
    _log = logging.getLogger('single_cell_biophysical.run_optimize')
    
    def __init__(self,
                 input_json,
                 output_json):
        self.input_json = input_json
        self.output_json = output_json
                
        #os.chdir(self.config_from_lims.lims_working_directory())

        #combined_manifest = 'optimize_manifest.json'
        
        self.app_config = Config().load('/home/timf/lims_sdk/manifest_lims_sdk.json')
        #self.manifest = self.config_from_lims.to_manifest(combined_manifest)
        self.manifest = self.app_config.manifest
        #self.input_config = self.config_from_lims  # TODO: verify this - might need copy for writing to lims upload json
        
        #app_config = Config()
        #self.description = app_config.load(combined_manifest)
        
        self.data_set = NwbDataSet(self.manifest.get_path('stimulus_path'))
    
    
    def nrnivmodl(self):
        RunOptimize._log.debug("nrnivmodl")
        
        sys.path.append('/local1/neuron/nrn/x86_64/bin')
        sys.path.append('/local1/neuron/iv/x86_64/bin')
        subprocess.call(['nrnivmodl', './modfiles'])
    
    
    def info(self):
        ''' return a string that a bash script can use
        to find the working directory, etc. to clean up.
        '''
        print(self.config_from_lims.lims_data['specimen']['id'])
        print(self.config_from_lims.lims_working_directory())
    
    
    def validate_config(self):
        specimen_config = self.config_from_lims.lims_data['specimen']
        
        if not 'specimen_tags' in specimen_config:
            raise(Exception('could not find speciment tags'))

    
    def copy_local(self):
        '''
        Note
        ----
        For files that aren't needed for local debugging, use write_manifest instead.
        '''
        modfile_dir = self.manifest.get_path('MODFILE_DIR')
            
        if not os.path.exists(modfile_dir):
            os.mkdir(modfile_dir)
        
        output_dir = self.manifest.get_path('WORKDIR')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)        
        
        modfiles = [self.manifest.get_path(key) for key,info
                    in self.manifest.path_info.items()
                    if 'format' in info and info['format'] == 'MODFILE']
        for from_file in modfiles:             
            RunOptimize._log.debug("copying %s to %s" % (from_file, modfile_dir))
            shutil.copy(from_file, modfile_dir)
        
        shutil.copy(self.manifest.get_path('fit_parameters_path'),
                    self.manifest.get_path('BASEDIR'))
        
        shutil.copy(self.manifest.get_path('stimulus_path'),
                    self.manifest.get_path('BASEDIR'))
    
    
    def write_manifest(self):
        '''
        Note
        ----
        Other necessary files are also written.
        '''
        shutil.copy(resource_filename(hoc_location.__name__,
                                      'cell.hoc'),
                    os.curdir)
        

        for p in self.config_from_lims.mod_file_paths():
            shutil.copy(p, 'modfiles')
    
    
    def start_specimen(self):
        import allensdk.model.biophysical.run_passive_fit as run_passive_fit
        import allensdk.model.biophysical.fit_stage_1 as fit_stage_1
        import allensdk.model.biophysical.fit_stage_2 as fit_stage_2
        
        self.passive_fit_data = \
            run_passive_fit.run_passive_fit(self.description)
        
        ju.write(os.path.join(self.config_from_lims.output_directory(),
                              'passive_fit_data.json'),
                 self.passive_fit_data)
        
        self.stage_1_jobs = \
            fit_stage_1.prepare_stage_1(self.description,
                                        self.passive_fit_data)
        
        ju.write(os.path.join(self.config_from_lims.output_directory(),
                              'stage_1_jobs.json'),
                 self.stage_1_jobs)
        fit_stage_1.run_stage_1(self.stage_1_jobs)
        
        output_directory = self.description.manifest.get_path('WORKDIR')
        stage_2_jobs = fit_stage_2.prepare_stage_2(output_directory)
        fit_stage_2.run_stage_2(stage_2_jobs)
    
    
    def make_fit(self):
        fit_types = ["f9", "f13"]
        
        best_fit_values = { fit_type: None for fit_type in fit_types }

        specimen_id = self.config_from_lims.lims_data['specimen']['id']
        
        for fit_type in fit_types:
            fit_type_dir = self.manifest.get_path('fit_type_path', fit_type)
            
            if os.path.exists(fit_type_dir):
                report = Report(self.description,
                                fit_type)
                report.generate_fit_file()
                if fit_type in best_fit_values.keys():               
                    best_fit_values[fit_type] = report.best_fit_value()
                
        best_fit_type, min_fit_value = reduce(lambda a, b: a if (a[1] < b[1]) else b,
                                              (i for i in best_fit_values.items() if i[1] is not None))        
        best_fit_file = self.manifest.get_path('output_fit_file',
                                               specimen_id,
                                               best_fit_type)
         
        self.input_config.add_well_known_file('application/json',
                                              best_fit_file,
                                              well_known_file_type_id=OptimizeConfigReader.NEURONAL_MODEL_PARAMETERS)
        self.input_config.write_file(self.lims_response_json)    
    


def main(command, input_json, output_json):
    ''' Entry point for module.
        :param command: select behavior, nrnivmodl or simulate
        :type command: string    
        :param lims_strategy_json: path to json file output from lims.
        :type lims_strategy_json: string
        :param lims_response_json: path to json file returned to lims.
        :type lims_response_json: string
    '''
    o = RunOptimize(input_json,
                    output_json)

    log_config = resource_filename('allensdk.model.biophysical',
                                    'logging.conf')
    lc.fileConfig(log_config)
    os.environ['LOG_CFG'] = log_config

    if 'nrnivmodl' == command:
        o.nrnivmodl()
    elif 'info' == command:
        o.info()
    elif 'write_manifest' == command:
        o.write_manifest()
        o.validate_config()                        
    elif 'copy_local' == command:
        o.copy_local()
    elif 'start_specimen' == command:
        o.start_specimen()
    elif 'make_fit' == command:
        o.make_fit()
    else:
        RunOptimize._log.error("no command")


if __name__ == '__main__':
    import sys
    
    command = sys.argv[-3]
    input_json = sys.argv[-2]
    output_json = sys.argv[-1]
    
    RunOptimize._log.debug("command: %s" % (command))
    RunOptimize._log.debug("input json: %s" % (input_json))
    RunOptimize._log.debug("output json: %s" % (output_json))

    main(command, input_json, output_json)
    
    RunOptimize._log.debug("success")