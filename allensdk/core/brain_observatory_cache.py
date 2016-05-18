import os
import allensdk.core.json_utilities as ju
from allensdk.api.cache import Cache
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi
from allensdk.config.manifest_builder import ManifestBuilder

class BrainObservatoryCache(Cache):
    EXPERIMENT_CONTAINERS_KEY = 'EXPERIMENT_CONTAINERS'
    EXPERIMENTS_KEY = 'EXPERIMENTS'
    EXPERIMENT_DATA_KEY = 'EXPERIMENT_DATA'

    def __init__(self, cache=True, manifest_file='brain_observatory_manifest.json', base_uri=None):
        super(BrainObservatoryCache, self).__init__(manifest=manifest_file, cache=cache)
        self.api = BrainObservatoryApi(base_uri=base_uri)
        

    def get_experiment_containers(self, file_name=None, targeted_structures=None, imaging_depths=None, transgenic_lines=None):
        file_name = self.get_cache_path(file_name, self.EXPERIMENT_CONTAINERS_KEY)

        if os.path.exists(file_name):
            containers = ju.read(file_name)
        else:
            containers = self.api.get_experiment_containers()

            if self.cache:
                ju.write(file_name, containers)

        return self.api.filter_experiment_containers(containers, targeted_structures, imaging_depths, transgenic_lines)


    def get_ophys_experiments(self, file_name=None, experiment_container_ids=None):
        file_name = self.get_cache_path(file_name, self.EXPERIMENTS_KEY)

        if os.path.exists(file_name):
            exps = ju.read(file_name)
        else:
            exps = self.api.get_ophys_experiments()

            if self.cache:
                ju.write(file_name, exps)

        return self.api.filter_ophys_experiments(exps, experiment_container_ids)

    
    def build_manifest(self, file_name):
        """
        Construct a manifest for this Cache class and save it in a file.
        
        Parameters
        ----------
        
        file_name: string
            File location to save the manifest.

        """

        mb = ManifestBuilder()

        mb.add_path('BASEDIR', '.')
        mb.add_path(self.EXPERIMENT_CONTAINERS_KEY, 'experiment_containers.csv', typename='file', parent_key='BASEDIR')
        mb.add_path(self.EXPERIMENTS_KEY, 'ophys_experiments.csv', typename='file', parent_key='BASEDIR')
        mb.add_path(self.EXPERIMENT_DATA_KEY, 'nwb_files/ophys_experiment_%d.nwb', typename='file', parent_key='BASEDIR')

        mb.write_json_file(file_name)


def main():
    from allensdk.api.api import Api
    import pandas as pd
        
    host = 'http://testwarehouse:9000'
    Api.default_api_url = host
    boc = BrainObservatoryCache()

    ecs = boc.get_experiment_containers(imaging_depths=[350])
    print "depth 350 experiment containers", len(ecs)

    ecs = boc.get_experiment_containers(targeted_structures=['VISp'])
    print "visp experiment containers", len(ecs)

    ecs = boc.get_experiment_containers(transgenic_lines=['Cux2-CreERT2'])
    print "cux2 experiment containers", len(ecs)

    exps = boc.get_ophys_experiments()
    print len(exps)
if __name__ == "__main__": main()
