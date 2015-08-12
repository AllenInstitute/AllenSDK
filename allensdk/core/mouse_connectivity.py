from allensdk.api.cache import Cache
from allensdk.config.model.manifest_builder import ManifestBuilder
from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
from allensdk.api.queries.structure.ontologies_api import OntologiesApi

import allensdk.core.json_utilities as json_utilities
from allensdk.core.ontology import Ontology

import nrrd, os
import pandas as pd
import numpy as np

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


    def get_experiments(self, dataframe=False, file_name=None, cre=None, injection_structure_ids=None):
        file_name = self.get_cache_path(file_name, 'EXPERIMENTS')

        if os.path.exists(file_name):
            experiments = json_utilities.read(file_name)
        else:
            experiments = self.api.experiment_source_search(injection_structures='root')            

            if self.cache:
                self.safe_mkdir(os.path.dirname(file_name))

                json_utilities.write(file_name, experiments)

        experiments = self.filter_experiments(experiments, cre, injection_structure_ids)

        if dataframe:
            experiments = pd.DataFrame(experiments)
            experiments.set_index(['id'], inplace=True, drop=False)

        return experiments


    def injection_in_structures(self, injection_structure_ids, query_injection_structure_ids, ontology):
        for injection_structure_id in injection_structure_ids:
            for query_injection_structure_id in query_injection_structure_ids:
                if ontology.structure_descends_from(injection_structure_id, query_injection_structure_id):
                    return True

        return False


    def filter_experiments(self, experiments, cre=None, injection_structure_ids=None):
        if cre == True:
            experiments = [ e for e in experiments if e['transgenic-line'] ]
        elif cre == False:
            experiments = [ e for e in experiments if not e['transgenic-line'] ]

        if injection_structure_ids is not None:
            ont = self.get_ontology()
            experiments = [ e for e in experiments if self.injection_in_structures([e['structure-id']], injection_structure_ids, ont) ]
                
        return experiments


    def get_experiment_structure_unionizes(self, experiment_id, file_name=None, is_injection=None, structure_ids=None, hemisphere_ids=None):
        file_name = self.get_cache_path(file_name, 'STRUCTURE_UNIONIZES', experiment_id)

        if os.path.exists(file_name):
            unionizes = pd.DataFrame.from_csv(file_name)
        else:
            unionizes = self.api.get_structure_unionizes([experiment_id])
            unionizes = pd.DataFrame(unionizes)

            # rename section_data_set_id column to experiment_id
            unionizes.columns = [ 'experiment_id' 
                                  if c == 'section_data_set_id' else c
                                  for c in unionizes.columns ]
                
            if self.cache:
                self.safe_mkdir(os.path.dirname(file_name))

                unionizes.to_csv(file_name)

        return self.filter_structure_unionizes(unionizes, is_injection, structure_ids, hemisphere_ids)


    def filter_structure_unionizes(self, unionizes, is_injection=None, structure_ids=None, hemisphere_ids=None):
        if is_injection is not None:
            unionizes = unionizes[unionizes.is_injection == is_injection]

        if structure_ids is not None:
            unionizes = unionizes[unionizes['structure_id'].isin(structure_ids)]
                                  
        if hemisphere_ids is not None:
            unionizes = unionizes[unionizes['hemisphere_id'].isin(hemisphere_ids)]
                                  
        return unionizes
        

    def get_structure_unionizes(self, experiment_ids, is_injection=None, structure_ids=None, hemisphere_ids=None):
        unionizes = [ self.get_experiment_structure_unionizes(eid, 
                                                              is_injection=is_injection, 
                                                              structure_ids=structure_ids, 
                                                              hemisphere_ids=hemisphere_ids) 
                      for eid in experiment_ids ]

        return pd.concat(unionizes, ignore_index = True)
            
        
    def get_structure_mask(self, structure_id, file_name=None, annotation_file_name=None):
        file_name = self.get_cache_path(file_name, 'STRUCTURE_MASK', structure_id)
        
        if os.path.exists(file_name):
            return nrrd.read(file_name)
        else:
            ont = self.get_ontology()
            structure_ids = ont.get_descendants(structure_id)
            annotation, _ = self.get_annotation_volume(annotation_file_name)
            mask = self.make_structure_mask(structure_ids, annotation)
            
            if self.cache:
                nrrd.write(file_name, mask)

            return mask


    def make_structure_mask(self, structure_ids, annotation):
        m = np.zeros(annotation.shape, dtype=np.uint8)

        for i,sid in enumerate(structure_ids):
            m[annotation==sid] = 1
        
        return m

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

