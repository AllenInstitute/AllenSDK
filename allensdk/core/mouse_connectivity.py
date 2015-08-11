from allensdk.api.cache import Cache
from allensdk.config.model.manifest_builder import ManifestBuilder
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.api.queries.structure.ontologies_api import OntologiesApi

import allensdk.core.json_utilities as json_utilities
from allensdk.core.ontology import Ontology

import nrrd, os
import pandas as pd

class MouseConnectivity(Cache):
    def __init__(self, resolution=25, cache=True, manifest_file='manifest.json'):
        super(MouseConnectivity, self).__init__(manifest=manifest_file, cache=cache)

        self.resolution = resolution
        self.api = MouseConnectivityApi()


    def get_annotation_volume(self, file_name=None):
        file_name = self.get_cache_path(file_name, 'ANNOTATION', self.resolution)

        if file_name is None:
            raise Exception("No save file provided for annotation volume.")

        if os.path.exists(file_name):
            annotation, info = nrrd.read(file_name)
        else:
            self.safe_mkdir(os.path.dirname(file_name))

            annotation, info = self.api.get_annotation_volume(self.resolution, file_name)

        return annotation, info

    
    def get_projection_density(self, experiment_id, file_name=None):  
        
        file_name = self.get_cache_path(file_name, 'PROJECTION_DENSITY', experiment_id, self.resolution)
        
        if file_name is None:
            raise Exception("No file name to save volume.")

        if not os.path.exists(file_name):
            self.safe_mkdir(os.path.dirname(file_name))
            self.api.download_projection_density(file_name, experiment_id, self.resolution)
                                              
        return nrrd.read(file_name)


    def get_injection_density(self, experiment_id, file_name=None):        
        file_name = self.get_cache_path(file_name, 'INJECTION_DENSITY', experiment_id, self.resolution)
        
        if file_name is None:
            raise Exception("No file name to save volume.")

        if not os.path.exists(file_name):
            self.safe_mkdir(os.path.dirname(file_name))

            self.api.download_injection_density(file_name, experiment_id,  self.resolution)
                                              
        return nrrd.read(file_name)


    def get_injection_fraction(self, experiment_id, file_name=None):        
        file_name = self.get_cache_path(file_name, 'INJECTION_FRACTION', experiment_id, self.resolution)
        
        if file_name is None:
            raise Exception("No file name to save volume.")

        if not os.path.exists(file_name):
            self.safe_mkdir(os.path.dirname(file_name))

            self.api.download_injection_fraction(file_name, experiment_id, self.resolution)
                                              
        return nrrd.read(file_name)


    def get_data_mask(self, experiment_id, file_name=None):        
        file_name = self.get_cache_path(file_name, 'DATA_MASK', experiment_id, self.resolution)
        
        if file_name is None:
            raise Exception("No file name to save volume.")

        if not os.path.exists(file_name):
            self.safe_mkdir(os.path.dirname(file_name))

            self.api.download_data_mask(file_name, experiment_id, self.resolution)
                                              
        return nrrd.read(file_name)


    def get_ontology(self, file_name=None):
        return Ontology(self.get_structures(file_name))


    def get_structures(self, file_name=None):
        file_name = self.get_cache_path(file_name, 'STRUCTURES')
        
        if os.path.exists(file_name):
            structures = pd.DataFrame.from_csv(file_name)
        else:
            structures = OntologiesApi().get_structures(1)
            structures = pd.DataFrame(structures)

            if self.cache:
                self.safe_mkdir(os.path.dirname(file_name))

                structures.to_csv(file_name)

        structures.set_index(['id'], inplace=True, drop=False)
        return structures


    def get_experiments(self, dataframe=False, file_name=None):
        file_name = self.get_cache_path(file_name, 'EXPERIMENTS')

        if os.path.exists(file_name):
            experiments = json_utilities.read(file_name)
        else:
            experiments = self.api.get_experiments(None,
                                                   include='specimen(donor(transgenic_lines))',
                                                   num_rows='all',
                                                   count=False) 
            
            if self.cache:
                self.safe_mkdir(os.path.dirname(file_name))

                json_utilities.write(file_name, experiments)

        if dataframe:
            experiments = pd.DataFrame(experiments)
            experiments.set_index(['id'], inplace=True)

        return experiments


    def get_structure_unionizes(self, experiment_id, file_name=None):
        file_name = self.get_cache_path(file_name, 'STRUCTURE_UNIONIZES', experiment_id)
        
        if os.path.exists(file_name):
            unionizes = pd.DataFrame.from_csv(file_name)
        else:
            unionizes = self.api.get_structure_unionizes([experiment_id], is_injection=None)
            unionizes = pd.DataFrame(unionizes)

            # rename section_data_set_id column to experiment_id
            unionizes.columns = [ 'experiment_id' 
                                  if c == 'section_data_set_id' else c
                                  for c in unionizes.columns ]
            unionizes.set_index(['id'], inplace=True)
                
            if self.cache:
                self.safe_mkdir(os.path.dirname(file_name))

                unionizes.to_csv(file_name)

        return unionizes
        

    def build_manifest(self, file_name):
        manifest_builder = ManifestBuilder()      
        manifest_builder.add_path('BASEDIR', '.')

        manifest_builder.add_path('EXPERIMENTS',
                                  'experiments.json',
                                  parent_key='BASEDIR',
                                  typename='file')
        
        manifest_builder.add_path('STRUCTURES',
                                  'structures.csv',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.add_path('STRUCTURE_UNIONIZES',
                                  'experiment_%d/structure_unionizes.csv',
                                  parent_key='BASEDIR',
                                  typename='file')
        
        manifest_builder.add_path('ANNOTATION',
                                  'annotation_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')
        
        manifest_builder.add_path('INJECTION_DENSITY',
                                  'experiment_%d/injection_density_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')
        
        manifest_builder.add_path('INJECTION_FRACTION',
                                  'experiment_%d/injection_fraction_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')
        
        manifest_builder.add_path('DATA_MASK',
                                  'experiment_%d/data_mask_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')
        
        manifest_builder.add_path('PROJECTION_DENSITY',
                                  'experiment_%d/projection_density_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')
        
        manifest_builder.add_path('STRUCTURE_MASK',
                                  'structure_masks/structure_%d.nrrd',
                                  parent_key='BASEDIR',
                                  typename='file')

        manifest_builder.write_json_file(file_name)

