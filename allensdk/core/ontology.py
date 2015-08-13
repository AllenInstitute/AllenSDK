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


    def __getitem__(self, structures):
        """ 
        Return a subset of structures by id or acronym. Duplicate values are ignored.

        Parameters
        ----------

        structures: tuple
            Elements can be pandas.Series objects, which are expected to be structure ids.
            Elements can be strings, which are expected to be acronyms.
            All other elements must be cast-able to int, which are treated as structure ids.

        Returns
        -------
        
        pandas.DataFrame
            A subset of rows from the complete ontology that match filtering criteria.
        """

        # __getitem__ always has a single argument.  If called with a single argument 
        # (e.g. ontology[315]), that item is passed straight through.  If called with 
        # multiple arguments (e.g. ontology[315,997]), that gets passed through as a
        # tuple.  This normalizes the arguments so that everything is a tuple.
        if not isinstance(structures, tuple):
            structures = structures,

        # this is the final set of structure ids used to filter
        structure_ids = set()

        string_strs = []
        for s in structures:
            if isinstance(s, pd.Series):
                # if it's a pandas series, assume it's a series of structure ids
                structure_ids.update(s.tolist())
            elif isinstance(s, str):
                # if it's a string, assume it's an acronym
                string_strs.append(s)
            else:
                # if it's anything else, cast it to an integer and treat it like a structure id
                structure_ids.add(int(s))

        # convert the string arguments to rows
        if len(string_strs):

            # pull out the rows that match these acronyms
            string_strs = self.df[self.df['acronym'].isin(string_strs)]

            # if there are no other structure ids, just return this dataframe
            if len(structure_ids) == 0:
                return string_strs
                
            # otherwise pull out the ids and add them to the set
            structure_ids.update(string_strs.id.tolist())

        print structure_ids
        return self.df.loc[structure_ids].dropna(axis=0,how='all')


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
