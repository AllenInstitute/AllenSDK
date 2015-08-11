from collections import defaultdict

import numpy as np
import pandas as pd
import allensdk.core.json_utilities as json_utilities

class Ontology( object ):      
    def __init__(self, df):
        self.df = df

        self.children = defaultdict(set)
        self.descendants = defaultdict(set)

        for i,s in df.iterrows():
            parent_id = s['parent_structure_id']
            if np.isfinite(parent_id):
                parent_id = int(parent_id)
                self.children[parent_id].add(s['id'])

            parent_id_list = map(int, s['structure_id_path'].split('/')[1:-1])

            for parent_id in parent_id_list:
                self.descendants[parent_id].add(s['id'])

    def __getitem__(self, structure_id):
        return self.df.loc[structure_id]

    def get_structure_by_acronym(self, acronym):
        pass

    def get_descendants(self, structure_id=None):
        if structure_id is None:
            return self.descendants
        else:
            return self.descendants[structure_id]

    def get_children(self, structure_id=None):
        if structure_id is None:
            return self.children
        else:
            return self.children[structure_id]

    @staticmethod
    def from_csv(csv_file):
        return Ontology(pd.DataFrame.from_csv(csv_file))

    @staticmethod
    def from_json(json_file):
        structures = json_utilities.read(json_file)
        df = pd.DataFrame(structures)
        return Ontology(df)
