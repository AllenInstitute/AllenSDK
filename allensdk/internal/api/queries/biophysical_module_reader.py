import json, os, logging

import allensdk.core.lims_utilities as lims_utilities
from allensdk.config.manifest_builder import ManifestBuilder
from allensdk.config.manifest import Manifest

class BiophysicalModuleReader(object):
    STIMULUS_CONTENT_TYPE = None
    MORPHOLOGY_TYPE_ID = 303941301
    MOD_FILE_TYPE_ID = 292178729
        
    def __init__(self):
        self._log = logging.getLogger(__name__)
        self.lims_path = None
        self.lims_data = None
        self.lims_update_data = None
    
    def read_lims_message(self, message, lims_path):
        self.lims_path = lims_path
        self.lims_data = message[0]
        self.lims_update_data = dict(self.lims_data)
    
    def read_lims_file(self, lims_path):
        self.lims_path = lims_path
        self.read_json(lims_path)
        self.lims_update_data = dict(self.lims_data)
    
    
    def read_json(self, path):
        with open(path, 'rb') as f:
            self.read_json_string(f.read())
    
    
    def read_json_string(self, json_string):
        self.lims_data = json.loads(json_string)
        self.lims_update_data = dict(self.lims_data)
    
    
    def stimulus_file_entries(self):
        ''' read the well known file path from the lims result
            corresponding to the stimulus file
            :return: well_known_file entries
            :rtype: array of dicts
        '''
        neuronal_model = self.lims_data['neuronal_model']
        specimen = neuronal_model['specimen']
        roi_result = specimen['ephys_roi_result']
        well_known_files = roi_result['well_known_files']
        
        stimulus_file_entries = []
        
        for well_known_file in well_known_files:
            try:
                file_type_id = well_known_file['well_known_file_type_id']
                
                if file_type_id == lims_utilities.NWB_FILE_TYPE_ID:
                    stimulus_file_entries.append(well_known_file)
            except:
                self._log.warn('skipping well known file record with no well known file type.')
        
        return stimulus_file_entries
    
    
    def stimulus_path(self):
        ''' Get the path to the stimulus file from the lims result.
            :return: path to stimulus file
            :rtype: string
        '''
        file_entries = self.stimulus_file_entries()
        
        if len(file_entries) > 1:
            self._log.warning('More than one stimulus file found.')
        
        file_entry = file_entries[0]
        
        stimulus_path = os.path.join(file_entry['storage_directory'],
                                     file_entry['filename'])
        
        return stimulus_path
    
    
    def lims_working_directory(self):
        ''' While this is the same directory as the neuronal_model_run
            directory, it can be mocked out for testing if the
            other directory is read only.
        '''
        return self.neuronal_model_run_dir()    
    
    def neuronal_model_run_dir(self):
        ''' read the directory path where
            output goes from the lims optimization config json
            
            Parameters
            ----------
            
            Returns
            -------
            string:
                directory path
        '''
        return self.lims_data['storage_directory']
    
    
    def fit_parameters_file_entries(self):
        ''' read the fit_parameter file path from the lims result
            corresponding to the stimulus file
            :return: well_known_file entries
            :rtype: array of dicts
        '''
        neuronal_model = self.lims_data['neuronal_model']
        well_known_files = neuronal_model['well_known_files']
        
        fit_parameter_file_entries = []
                
        for well_known_file in well_known_files:
            file_type_id = well_known_file['well_known_file_type_id']
            
            if file_type_id == lims_utilities.MODEL_PARAMETERS_FILE_TYPE_ID:
                fit_parameter_file_entries.append(well_known_file)
                
        return fit_parameter_file_entries
    
    
    def fit_parameters_path(self):
        ''' Get the path to the fit parameters file from the lims result.
            :return: path to file
            :rtype: string
        '''
        file_entries = self.fit_parameters_file_entries()
        
        if len(file_entries) > 1:
            self._log.warning('More than one fit parameter file found.')
        
        file_entry = file_entries[0]
        
        fit_parameter_path = os.path.join(file_entry['storage_directory'],
                                          file_entry['filename'])
 
        return fit_parameter_path
    
    def model_type(self):
        ''' TODO: comment
        '''
        return self.lims_data['neuronal_model']['neuronal_model_template']['name']
    
    def morphology_file_entries(self):
        ''' read the well known file paths
            from the lims result corresponding to the morphology
            
            Returns
            -------
            arrary of dicts:
                well known file entries
        '''
        neuronal_model = self.lims_data['neuronal_model']
        specimen = neuronal_model['specimen']
        reconstructions = specimen['neuron_reconstructions']
        
        morphology_file_entries = []
        
        for reconstruction in reconstructions:
            superseded = reconstruction['superseded']
            manual = reconstruction['manual']
            
            if manual == True and superseded == False:
                well_known_files = reconstruction['well_known_files']
                
                for well_known_file in well_known_files:
                    file_type_id = well_known_file['well_known_file_type_id']
                    
                    if file_type_id == BiophysicalModuleReader.MORPHOLOGY_TYPE_ID:
                        morphology_file_entries.append(well_known_file)
        
        return morphology_file_entries
    
    
    def morphology_path(self):
        ''' Get the path to the morphology file from the lims result.
            :return: path to morphology file
            :rtype: string
        '''
        file_entries = self.morphology_file_entries()
        
        if len(file_entries) > 1:
            self._log.warning('More than one morphology file found.')
        
        file_entry = file_entries[0]
        
        morphology_path = os.path.join(file_entry['storage_directory'],
                                     file_entry['filename'])
        
        return morphology_path
    
    
    def sweep_entries(self):
        ''' read the sweep entries
            from the lims result corresponding to the stimulus
            :return: stimulus sweep entries
            :rtype: array of dicts
        '''
        neuronal_model = self.lims_data['neuronal_model']
        specimen = neuronal_model['specimen']
        sweeps = specimen['ephys_sweeps']
        
        return sweeps
    
    
    def sweep_numbers_by_type(self):
        sweeps = self.sweep_entries()
        
        d = {s['ephys_stimulus']['ephys_stimulus_type']['name']: [] for s in sweeps}        
        
        for n, s in enumerate(sweeps):
            t = s['ephys_stimulus']['ephys_stimulus_type']['name']
            d[t].append(n)
        
        return d
      
    
    def sweep_numbers(self):
        ''' Get the stimulus sweep numbers from the lims result
            :return: list of sweep numbers
            :rtype: array of ints
        '''
        sweep_entries = self.sweep_entries()
        
        if not sweep_entries or len(sweep_entries) < 1:
            self._log.warning('No sweeps found.')
        
        sweeps = [sweep_entry['sweep_number'] \
                  for sweep_entry in sweep_entries \
                  if sweep_entry['workflow_state'] == 'auto_passed' or \
                     sweep_entry['workflow_state'] == 'manual_passed' ]
        
        return list(set(sweeps))
    
    
    def mod_file_entries(self):
        ''' read the NERUON .mod file entries
            from the lims result corresponding to the NeuronModel
            :return: well known file entries
            :rtype: array of dicts
        '''
        neuronal_model = self.lims_data['neuronal_model']
        model_template = neuronal_model['neuronal_model_template']
        well_known_files = model_template['well_known_files']
        
        mod_file_entries = []
        
        for well_known_file in well_known_files:
            file_type_id = well_known_file['well_known_file_type_id']
                            
            if file_type_id == BiophysicalModuleReader.MOD_FILE_TYPE_ID:
                mod_file_entries.append(well_known_file)
                
        return mod_file_entries
    
    
    def mod_file_paths(self):
        ''' Get the paths to the mod files from the lims result.
            :return: paths to mod files
            :rtype: array of strings
        '''
        file_entries = self.mod_file_entries()
        
        if not file_entries or len(file_entries) < 1:
            self._log.warning('No mod files found.')
        
        mod_file_paths = []
        
        for file_entry in file_entries:
            mod_path = os.path.join(file_entry['storage_directory'],
                                    file_entry['filename'])
            self._log.info(mod_path)
            mod_file_paths.append(mod_path)
        
        return mod_file_paths
    
    
    def add_well_known_file(self, content_type, path):
        if path in self.result_paths():
            self._log.warn("Previously existing result file %s, not adding new well-known-file relation." % (path))
        
        (dirname, filename) = os.path.split(os.path.abspath(path))
        self.lims_update_data['well_known_files'] += [{
                'content_type': content_type,
                'filename': filename,
                'storage_directory': dirname
            }]
    
    
    def set_workflow_state(self, state):
        self.lims_update_data['workflow_state'] = 'success'
    
    
    def result_file_entries(self):
        ''' read previously existing well known file entries
            from the neuronal model run entry
            corresponding to a result file from a previous run
            :return: well_known_file entries
            :rtype: array of dicts
        '''
        well_known_files = self.lims_data['well_known_files']
        result_file_entries = []
        
        for well_known_file in well_known_files:
            try:
                file_type_id = well_known_file['well_known_file_type_idn']
                
                if file_type_id == lims_utilities.NWB_FILE_TYPE_ID:
                    result_file_entries.append(well_known_file)
            except:
                self._log.warn('skipping well known file record with no well known file type.')
        
        return result_file_entries
    
    
    def result_paths(self):
        ''' Get the path to a previously existing result file from the lims result.
            :return: paths to previously existing result files
            :rtype: list of strings
        '''
        file_entries = self.result_file_entries()
        
        if len(file_entries) > 0:
            self._log.warning('Previously existing result file found.')
        
        result_paths = []
        
        for file_entry in file_entries:
            result_paths.append(os.path.join(file_entry['storage_directory'],
                                             file_entry['filename']))
        
        self._log.info("Result paths; %s" % (result_paths))
        return result_paths
    
    
    def write_file(self, path):
        with open(path, 'wb') as f:
            f.write(json.dumps(self.lims_update_data, indent=2))
    
    
    def to_manifest(self, manifest_path=None):
        b = ManifestBuilder()
        b.add_path('BASEDIR', os.path.realpath(os.curdir))
        b.add_path('WORKDIR', 'workdir')
        b.add_path('MORPHOLOGY',
                       self.morphology_path(),
                       typename='file')
        b.add_path('CODE_DIR', 'templates')
        b.add_path('MODFILE_DIR', 'modfiles')
        
        for modfile in self.mod_file_entries():
            b.add_path('MOD_FILE_%s' % (os.path.splitext(modfile['filename'])[0]),
                       os.path.join(modfile['storage_directory'],
                                    modfile['filename']),
                       typename='file',
                       format='MODFILE')
        
        b.add_path('neuronal_model_run_data',
                   self.lims_path,
                   typename='file')
        b.add_path('stimulus_path',
                   self.stimulus_path(),
                   typename='file',
                   format='NWB')
        b.add_path('output_path',
                   os.path.basename(self.stimulus_path()),
                   typename='file',
                   parent_key='WORKDIR',
                   format='NWB')
        b.add_path('fit_parameters',
                   self.fit_parameters_path())
        
        b.add_section('bps_cfg',
                      {"biophys": [{"model_file": [ manifest_path,
                                    self.fit_parameters_path() ],
                                    "model_type": self.model_type()}]})
        
        b.add_section('stimulus_conf',
                      {"runs": [{"sweeps": self.sweep_numbers(),
                                 "sweeps_by_type": self.sweep_numbers_by_type()
                                 }]})
        
        b.add_section('hoc_conf',
                      {"neuron" : [{"hoc": [ "stdgui.hoc", "import3d.hoc" ]
                                    }]}) 
        
        
        m = Manifest(config=b.path_info)
        
        if manifest_path != None:
            b.write_json_file(manifest_path)
        
        return m
