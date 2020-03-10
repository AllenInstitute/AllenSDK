from allensdk.model.biophysical import runner,_schemas
import json
import os
from argschema import ArgSchemaParser
from allensdk.api.queries.biophysical_api import BiophysicalApi
from allensdk.core.nwb_data_set import NwbDataSet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

sns.set(style='whitegrid',font_scale=1.5)
plt.rcParams.update({'axes.grid':False})

#%% Define utility functions

def get_manifest_args(args):
    return runner.load_description(args)
    

def get_sweep_data(nwb_file,sweep_number):
    nwb = NwbDataSet(nwb_file)
    sweep = nwb.get_sweep(sweep_number)
    stim_diff = np.diff(sweep['stimulus']*1e12)
    stim_start = np.where(stim_diff != 0)[0][-2]
    stim_end = np.where(stim_diff != 0)[0][-1]
    
    # read v and t as numpy arrays
    v = sweep['response']*1e3
    dt = 1.0e3 / sweep['sampling_rate']
    num_samples = len(v)
    t = np.arange(num_samples) * dt
    return t,v,t[stim_start],t[stim_end]

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
            
#%% Pick a cell and a sweep to run from the available set of protocols
            
cell_id = 468193142 # get this from the web site: http://celltypes.brain-map.org
sweep_num = 46 # Select a Long Square sweep :  1s DC

#%% Download all-acive model

sdk_model_templates = {491455321:'all_active',329230710:'perisomatic'}
bp = BiophysicalApi()
bp.cache_stimulus = True 
model_list = bp.get_neuronal_models(cell_id)
model_dict = {}
for model_dict_ in model_list:
    if model_dict_['neuronal_model_template_id'] in sdk_model_templates.keys():
        model_dict[sdk_model_templates[model_dict_['neuronal_model_template_id']]] = model_dict_['id']

model_dir = 'all_active_models'
bp.cache_data(model_dict['all_active'], working_directory=model_dir)
new_model_file = 'fit_parameters_new.json'
shutil.copyfile(new_model_file,os.path.join(model_dir,new_model_file))
copytree('modfiles',os.path.join(model_dir,'modfiles'))

#%% Running the legacy all-active models

os.chdir(model_dir)
os.system('nrnivmodl modfiles/')
manifest_file = 'manifest.json'
manifest_dict = json.load(open(manifest_file))

# sweeps by type is not populated when the model is downloaded using the api
# in that case add the sweep type to the manifest 
if 'sweeps_by_type' not in manifest_dict['runs'][0]:
    manifest_dict['runs'][0]['sweeps_by_type'] = {"Long Square":[sweep_num]}
json.dump(manifest_dict,open(manifest_file,'w'),indent=2) 

schema_legacy = ArgSchemaParser(input_data=dict(manifest_file=manifest_file,log_level = 'INFO'),
                                         schema_type=_schemas.runner_config)
runner.run(schema_legacy.args,procs=1,sweeps=[sweep_num])

#%% Running the new all-active models

manifest_dict = json.load(open(manifest_file))

# Change the simulation output directory to avoid overwriting for the new models
for manifest_config in manifest_dict['manifest']:
    if manifest_config['key'] == 'WORKDIR':
        manifest_config['spec'] = 'work_new'

new_manifest_file = 'manifest_new.json' 
     
manifest_dict['biophys'][0]['model_file'] = [new_manifest_file,new_model_file]
json.dump(manifest_dict,open(new_manifest_file,'w'),indent=2)  

schema_new = ArgSchemaParser(input_data=dict(manifest_file=new_manifest_file,log_level = 'INFO', axon_type = 'stub_axon'),
                                         schema_type=_schemas.runner_config)
runner.run(schema_new.args,procs=1,sweeps=[sweep_num])

#%% Comparing the responses

legacy_config = get_manifest_args(schema_legacy.args)
output_nwb_legacy = legacy_config.manifest.get_path('output_path')

new_config =  get_manifest_args(schema_new.args)
output_nwb_new = new_config.manifest.get_path('output_path')

exp_nwb = new_config.manifest.get_path('stimulus_path')

t_exp,v_exp,stim_start,stim_end = get_sweep_data(exp_nwb,sweep_num)
t_legacy_aa,v_legacy_aa,_,_ = get_sweep_data(output_nwb_legacy,sweep_num)
t_new_aa,v_new_aa,_,_ = get_sweep_data(output_nwb_new,sweep_num)

fig,ax = plt.subplots(figsize=(10,6))
ax.plot(t_exp,v_exp, color = 'k', label='Experiment')
ax.plot(t_legacy_aa,v_legacy_aa,color = 'r', label= 'Legacy Model')
ax.plot(t_new_aa,v_new_aa,color='b', label = 'New Model')
ax.set_xlim([stim_start-200,stim_end+200])
sns.despine(ax=ax)
h,l = ax.get_legend_handles_labels()
fig.legend(h,l,ncol=3,frameon=False,loc='lower center')
fig.tight_layout(rect=[0.0,0.05,1,0.95])

plt.show()
