#Import standard modules.
import mpi4py as MPI
import numpy as np
import csv
import pickle
import sys
#Import application specific modules.
from allensdk.model.biophysical_perisomatic.utils import Utils
from allensdk.model.biophys_sim.config import Config
from utils import Utils
from allensdk.api.queries.biophysical_perisomatic_api import \
    BiophysicalPerisomaticApi
from allensdk.api.queries.cell_types_api import CellTypesApi
import allensdk.core.swc as swc
import allensdk 
import os
from allensdk.core.nwb_data_set import NwbDataSet
from IPython import __main__
bp = BiophysicalPerisomaticApi('http://api.brain-map.org')
ct = CellTypesApi()
cells = ct.list_cells()
#Below is one way, I can create URL, queries and get the result in a local python dictionary.
def __main__():
    print 'main'
def query_all_neurons():
    """
    Request only files associated with a model.
    Return API queries about biophysical perisomatic_neurons.
    Returns a list of well known file keys, but not yet properly formed URLs.
    """
    instance=allensdk.api.api.Api('http://api.brain-map.org')
    returned=allensdk.api.api.Api.retrieve_parsed_json_over_http(instance,'http://api.brain-map.org/api/v2/data/query.json?criteria=model::NeuronalModel')
    bphys = [n for n in returned['msg'] if 'Biophysical' in n['name'] ]
    sid=[ b['id'] for b in bphys ]
    listwk = [ bp.get_well_known_file_ids(n) for n in sid ]
    return listwk

def retrieve_files(filename,wkfc,instance,working_directory=None):    
    """
    Arguments: the file name and the well known file code, an instance of the Allen Brain API object.
    Builds a well known file URL, and actually downloads and caches the files locally. 
    Files are cached on hard disk nothing to return, so this function lacks a return statement.
    """    
    wk=instance.construct_well_known_file_download_url(wkfc) 
    instance.retrieve_file_over_http(wk,filename)    
    

def get_all_files(instance,bphys):
    ''' 
    bphys is a big list of dictionaries that contains all of the different types of files you might be 
    interested in. phys needs to be iterated through in order to download all of the files, this is 
    with the statement for n in bphys inside each list comprehension.
    Note that calling retrieve_files iteratively inside the list comprehensions is sufficient to
    download all the requested files, and store them to the current working directory.
    '''
    [ retrieve_files(y,x,instance) for n in bphys for x,y in n['modfiles'].iteritems() ]
    [ retrieve_files(y,x,instance) for n in bphys for x,y in n['morphology'].iteritems() ]
    [ retrieve_files(y,x,instance) for n in bphys for x,y in n['fit'].iteritems() ]
    [ retrieve_files(y,x,instance) for n in bphys for x,y in n['stimulus'].iteritems() ]    
    for n in bphys:
        bp.create_manifest(str(n['fit'].values()[0]),
                           str(n['stimulus'].values()[0]),
                           str(n['morphology'].values()[0]),
                           [0,1,2,3,4,5])
        print mp.manifest
        manifest_path = os.path.join(os.getcwd(), str(n['morphology'].values()[0])+str(manifest.json))
        with open(manifest_path, 'wb') as f:
            f.write(json.dumps(mp.manifest, indent=2))

bphys = query_all_neurons()   
#Uncomment the get_all_files function call to download everything. 
#Only download everything if you have not downloaded everything already if you have
#not done already as the nwb files are very large and time consuming to download

#get_all_files(instance,bphys)

#If everything has been downloaded already uncomment the following three statements
instance=allensdk.api.api.Api('http://api.brain-map.org')
downloads = [ instance.construct_well_known_file_download_url(d) for d in bphys ]
working_directory='neuronal_model'

#bphys is a list containing a dictionary of dictionaries, the keys of the most nested dictionaries are the suffixs of well known files. 
d=instance.construct_well_known_file_download_url(files[0]) 
cached_file_path = os.path.join(working_directory, d)
instance.retrieve_file_over_http(d,cached_file_path) 
for d in downloads:
    cached_file_path = os.path.join(working_directory, d)
    instance.retrieve_file_over_http(d,cached_file_path) 
def read_local_json():
    #Above and below are the same. Above is online version, below is offline version that acts on local files.
    #Use the other methods in the biophysical_perisomatic_api to one by one do the things needed to cache whole models.
    f1= open('neuron_models_from_query_builder.json')
    information = allensdk.api.api.load(f1)
    return information
information=read_local_json()
bp.cache_data(395310469, working_directory='neuronal_model')
for i in information['msg']:    
    bp.cache_data(i['id'], working_directory='neuronal_model')

config = Config().load('config.json')
utils = Utils(config)
