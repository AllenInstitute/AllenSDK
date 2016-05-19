import os
import allensdk.core.json_utilities as ju
from allensdk.api.cache import Cache
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi
from allensdk.config.manifest_builder import ManifestBuilder
from allensdk.core.cam_nwb_data_set import CamNwbDataSet
import pandas as pd

class BrainObservatoryCache(Cache):
    EXPERIMENT_CONTAINERS_KEY = 'EXPERIMENT_CONTAINERS'
    EXPERIMENTS_KEY = 'EXPERIMENTS'
    CELL_SPECIMENS_KEY = 'CELL_SPECIMENS'
    EXPERIMENT_DATA_KEY = 'EXPERIMENT_DATA'
    STIMULUS_MAPPINGS_KEY = 'STIMULUS_MAPPINGS'

    def __init__(self, cache=True, manifest_file='brain_observatory_manifest.json', base_uri=None):
        super(BrainObservatoryCache, self).__init__(manifest=manifest_file, cache=cache)
        self.api = BrainObservatoryApi(base_uri=base_uri)
        

    def get_targeted_structures(self):
        containers = self.get_experiment_containers()
        targeted_structures = set([ c['targeted_structure']['acronym'] for c in containers])
        return sorted(list(targeted_structures))


    def get_transgenic_lines(self):
        containers = self.get_experiment_containers()
        transgenic_lines = set([ tl['name'] for c in containers for tl in c['specimen']['donor']['transgenic_lines']])
        return sorted(list(transgenic_lines))


    def get_stimulus_names(self):
        exps = self.get_ophys_experiments()
        stimulus_names = set([ exp['stimulus_name'] for exp in exps ])
        return sorted(list(stimulus_names))


    def get_experiment_containers(self, file_name=None, targeted_structures=None, imaging_depths=None, transgenic_lines=None):
        file_name = self.get_cache_path(file_name, self.EXPERIMENT_CONTAINERS_KEY)

        if os.path.exists(file_name):
            containers = ju.read(file_name)
        else:
            containers = self.api.get_experiment_containers()

            if self.cache:
                ju.write(file_name, containers)

        return self.api.filter_experiment_containers(containers, targeted_structures, imaging_depths, transgenic_lines)


    def get_ophys_experiments(self, file_name=None, experiment_container_ids=None,
                              targeted_structures=None, imaging_depths=None, transgenic_lines=None,
                              stimulus_names=None):
        file_name = self.get_cache_path(file_name, self.EXPERIMENTS_KEY)

        if os.path.exists(file_name):
            exps = ju.read(file_name)
        else:
            exps = self.api.get_ophys_experiments()

            if self.cache:
                ju.write(file_name, exps)

        return self.api.filter_ophys_experiments(exps, experiment_container_ids, targeted_structures, 
                                                 imaging_depths, transgenic_lines, stimulus_names)


    def get_stimulus_mappings(self, file_name=None):
        file_name = self.get_cache_path(file_name, self.STIMULUS_MAPPINGS_KEY)

        if os.path.exists(file_name):
            mappings = ju.read(file_name)
        else:
            mappings = self.api.get_stimulus_mappings()

            if self.cache:
                ju.write(file_name, mappings)

        return mappings


    def get_cell_specimens(self, file_name=None, ophys_experiment_ids=None):
        file_name = self.get_cache_path(file_name, self.CELL_SPECIMENS_KEY)

        if os.path.exists(file_name):
            cell_specimens = ju.read(file_name)
        else:
            cell_specimens = self.api.get_cell_metrics()

            if self.cache:
                ju.write(file_name, cell_specimens)

        # drop the thumbnail columns
        mappings = self.get_stimulus_mappings()
        thumbnails = [ m['item'] for m in mappings if m['item_type'] == 'T' and m['level'] == 'R']

        cell_specimens = self.api.filter_cell_specimens(cell_specimens)
        
        df = pd.DataFrame.from_dict(cell_specimens)
        df = df.drop(thumbnails, axis=1)

        return df

    
    def get_ophys_experiment_data(self, ophys_experiment_id, file_name=None):
        file_name = self.get_cache_path(file_name, self.EXPERIMENT_DATA_KEY, ophys_experiment_id)

        if not os.path.exists(file_name):
            self.api.save_ophys_experiment_data(ophys_experiment_id, file_name)

        return CamNwbDataSet(file_name)

    
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
        mb.add_path(self.EXPERIMENT_DATA_KEY, 'ophys_experiment_data/%d.nwb', typename='file', parent_key='BASEDIR')
        mb.add_path(self.CELL_SPECIMENS_KEY, 'cell_specimens.csv', typename='file', parent_key='BASEDIR')
        mb.add_path(self.STIMULUS_MAPPINGS_KEY, 'stimulus_mappings.csv', typename='file', parent_key='BASEDIR')

        mb.write_json_file(file_name)


def main():
    from allensdk.api.api import Api
    import pandas as pd
        
    host = 'http://testwarehouse:9000'
    Api.default_api_url = host
    boc = BrainObservatoryCache()

    ecs = boc.get_experiment_containers()
    print "all ecs", len(ecs)

    ecs = boc.get_experiment_containers(imaging_depths=[350])
    print "depth 350 experiment containers", len(ecs)

    ecs = boc.get_experiment_containers(targeted_structures=['VISp'])
    print "visp experiment containers", len(ecs)

    ecs = boc.get_experiment_containers(transgenic_lines=['Cux2-CreERT2'])
    print "cux2 experiment containers", len(ecs)

    exps = boc.get_ophys_experiments(experiment_container_ids=[ ec['id'] for ec in ecs ])
    print len(exps)

    ds = boc.get_ophys_experiment_data(exps[0]['id'])
if __name__ == "__main__": main()
