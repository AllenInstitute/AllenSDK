from collections import defaultdict

import numpy as np
import pandas as pd
import allensdk.core.json_utilities as json_utilities

class Ontology( object ):      
    def __init__(self, df):
        self.df = df

        child_ids = defaultdict(set)
        descendant_ids = defaultdict(set)

        for i,s in df.iterrows():
            parent_id = s['parent_structure_id']
            if np.isfinite(parent_id):
                parent_id = int(parent_id)
                child_ids[parent_id].add(s['id'])

            parent_id_list = map(int, s['structure_id_path'].split('/')[1:-1])

            for parent_id in parent_id_list:
                descendant_ids[parent_id].add(s['id'])

        self.child_ids = dict(child_ids)
        self.descendant_ids = dict(descendant_ids)


    def __getitem__(self, *structure_ids):
        try:
            return self.df.loc[structure_ids]
        except KeyError:
            return None


    def get_structure_by_acronym(self, acronym):
        return self.df[self.df['acronym'] == acronym]


    def get_descendant_ids(self, *structure_ids):
        if len(structure_ids) == 0:
            return self.descendant_ids
        else:
            descendants = set()
            for structure_id in structure_ids:
                descendants.update(self.descendant_ids.get(int(structure_id), set()))
            return descendants


    def get_child_ids(self, *structure_ids):
        if len(structure_ids) == 0:
            return self.child_ids
        else:
            children = set()
            for structure_id in structure_ids:
                children.update(self.child_ids.get(int(structure_id), set()))
            return children
                                

    def get_descendants(self, structure_id):
        descendant_ids = self.get_descendant_ids(structure_id)
        return self[descendant_ids]


    def get_children(self, structure_id):
        child_ids = self.child_ids(structure_id)
        return self[child_ids]


    def structure_descends_from(self, child_id, parent_id):
        child = self[child_id]

        if child is not None:
            parent_str = '/%d/' % parent_id
            return child['structure_id_path'].find(parent_str) >= 0
        
        return False


    @staticmethod
    def from_csv(csv_file):
        df = pd.DataFrame.from_csv(csv_file)
        df.set_index(['id'], inplace=True, drop=False)
        return Ontology(df)

    @staticmethod
    def from_json(json_file):
        structures = json_utilities.read(json_file)
        df = pd.DataFrame(structures)
        df.set_index(['id'], inplace=True, drop=False)
        return Ontology(df)
