# Copyright 2015 Allen Institute for Brain Science
# This file is part of Allen SDK.
#
# Allen SDK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Allen SDK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Allen SDK.  If not, see <http://www.gnu.org/licenses/>.

import re
import csv
import copy

# Default columns to be read and written by this module 
DEFAULT_COLUMNS = [ 'id', 'type', 'x', 'y', 'z', 'radius', 'parent' ]

# Default columns to convert to numeric types automatically 
DEFAULT_NUMERIC_COLUMNS = [ 'type', 'x', 'y', 'z', 'radius' ]

def read_swc(file_name, columns=None, numeric_columns=None):
    """  Read in an SWC file and return a Morphology object.
    SWC are basically CSV files, but they often don't have headers. 
    You can pass those in explicitly and also indicate which
    columns are numeric.

    Parameters
    ----------
    file_name: string
        SWC file name.

    columns: list of strings
        names of the columns in this file (default: DEFAULT_COLUMNS)

    numeric_columns: list of strings
        names of the numeric columns in this file (default: DEFAULT_NUMERIC_COLUMNS)

    Returns
    -------
    Morphology
        A Morphology instance.
    """

    if columns is None:
        columns = DEFAULT_COLUMNS
        
    if numeric_columns is None:
        numeric_columns = DEFAULT_NUMERIC_COLUMNS 

    with open(file_name, "rb") as f:
        # skip comment rows, strip off extra whitespace
        return read_rows(f, columns, numeric_columns)


def read_rows(rows, columns, numeric_columns):
    """ Parse a list of string SWC rows.  Lines that start with '#'
    are ignored.  Numeric types are properly converted.

    Parameters
    ----------
    rows: list of strings
        usually the rows of an SWC file

    columns: list of strings
        names of the columns in this file

    numeric_columns: list of strings
        names of the numeric columns in this file

    Returns
    -------
    Morphology
        A Morphology instance.
    """

    rows = [ r.strip() for r in rows if len(r) > 0 and r.strip()[0] != '#' ]
    
    reader = csv.DictReader(rows, fieldnames=columns, delimiter=' ', skipinitialspace=True, restkey='other')
    
    compartment_list = []

    # convert numeric columns 
    for compartment in reader:
        for nh in numeric_columns:
            compartment[nh] = str_to_num(compartment[nh])
                
        compartment_list.append(compartment)

    return Morphology(compartment_list=compartment_list)    
    

def read_string(s, columns=None, numeric_columns=None):
    """ Parse a list of string SWC rows.  Lines that start with '#'
    are ignored.  Numeric types are properly converted.

    Parameters
    ----------
    s: string
        the contents of an SWC file as a string

    columns: list of strings
        names of the columns in this file (default: DEFAULT_COLUMNS)

    numeric_columns: list of strings
        names of the numeric columns in this file (default: DEFAULT_NUMERIC_COLUMNS)

    Returns
    -------
    Morphology
        A Morphology instance.
    
    """
    if columns is None:
        columns = DEFAULT_COLUMNS
        
    if numeric_columns is None:
        numeric_columns = DEFAULT_NUMERIC_COLUMNS 

    rows = s.split('\n')

    return read_rows(rows, columns, numeric_columns)


class Morphology( object ):
    """ Keep track of the list of compartments in a morphology and provide 
    a few helper methods (index by id, sparsify, root, etc).  During initialization
    the compartments are assigned a 'children' property that is a list of
    pointers to child compartments.
    """

    SOMA = 1

    def __init__(self, compartment_list=None, compartment_index=None):
        """ Try to initialize from a list of compartments first, then from
        a dictionary indexed by compartment id if that fails, and finally just
        leave everything empty.

        Parameters
        ----------
        compartment_list: list 
            list of compartment dictionaries
            
        compartment_index: dict
            dictionary of compartments indexed by id
        """

        self._compartment_list = []
        self._compartment_index = {}
        
        # first try the compartment list, then try the compartment index
        if compartment_list:
            self.compartment_list = compartment_list
        elif compartment_index:
            self.compartment_index = compartment_index


    @property 
    def compartment_list(self):
        """ Return the compartment list.  This is a property to ensure that the 
        compartment list and compartment index are in sync. """
        return self._compartment_list



    @compartment_list.setter
    def compartment_list(self, compartment_list):
        """ Update the compartment list.  Update the compartment index. """
        self._compartment_list = compartment_list
        self._compartment_index = { c['id']: c for c in compartment_list }
        self.update_children()


    @property
    def compartment_index(self):
        """ Return the compartment index.  This is a property to ensure that the
        compartment list and compartment index are in sync. """
        return self._compartment_index


    @compartment_index.setter
    def compartment_index(self, compartment_index):
        """ Update the compartment index.  Update the compartment list. """
        self._compartment_index = compartment_index
        self._compartment_list = compartment_index.values()
        self.update_children()


    @property
    def root(self):
        """ Search through the compartment index for the root compartment """
        for cid,c in self.compartment_index.iteritems():
            if c['parent'] == '-1':
                return c
        return None


    def compartment_index_by_type(self, compartment_type):
        """ Return an dictionary of compartments indexed by id that all have
        a particular compartment type.

        Parameters
        ----------
        compartment_type: int
            Desired compartment type
        """

        return { c['id']: c for c in self._compartment_list if c['type'] == compartment_type }


    def write(self, file_name, columns=None):
        """ Write this morphology out to an SWC file 
      
        Parameters
        ----------
        file_name: string
            desired name of your SWC file

        columns: list
            columns to write to your SWC file (default: DEFAULT_COLUMNS)
        """
        if columns is None:
            columns = DEFAULT_COLUMNS
        
        # convert to array, sort by id
        sorted_compartments = sorted(self.compartment_list, key=lambda x: int(x['id']))

        with open(file_name, 'wb') as f:
            writer = csv.DictWriter(f, delimiter=' ', fieldnames=columns, extrasaction='ignore')
            writer.writerows(sorted_compartments)


    def validate(self):
        """ Make sure that the parents and children are assigned properly. """
        compartments = self._compartment_index

        for cid, c in self.compartment_index.iteritems():
            if c['parent'] != "-1":
                assert c['parent'] in compartments, "bad parent id: %s" % (c['parent'] )
                
            for child in c['children']:
                assert child in compartments, "bad child id: %s" % (child)

        
    def update_children(self):
        """ Fill each compartment's array of children """
        compartments = self._compartment_index

        for i,compartment in compartments.iteritems():
            compartment['children'] = []

        for i,compartment in compartments.iteritems():
            pi = compartment['parent']
    
            if pi not in compartments:
                continue 

            parent = compartments[pi]
            
            parent['children'].append(compartment['id'])


    def sparsify(self, modulo, compress_ids=False):
        """ Return a new Morphology object that has a given number of non-leaf,
        non-root segments removed.  IDs can be reassigned so as to be continuous.

        Parameters
        ----------
        modulo: int
           keep 1 out of every modulo segments.

        compress_ids: boolean
           Reassign ids so that ids are continuous (no missing id numbers).

        Returns
        -------   
        Morphology
            A new morphology instance
        """
        
        compartments = copy.deepcopy(self.compartment_index)
        root = self.root

        keep = {}

        # figure out which compartments to toss
        ct = 0
        for i, c in compartments.iteritems():
            pid = c['parent']
            cid = c['id']
            ctype = c['type']

            # keep the root, soma, junctions, and the first child of the root (for visualization)
            if pid == "-1" or len(c['children']) != 1 or pid == root['id'] or ctype == Morphology.SOMA:
                keep[cid] = True
            else:
                keep[cid] = (ct % modulo) == 0
                
            ct += 1

        # hook children up to their new parents
        for i, c in compartments.iteritems():
            comp_id = c['id']

            if keep[comp_id] is False:
                parent_id = c['parent']
                while keep[parent_id] is False:
                    parent_id = compartments[parent_id]['parent']

                for child_id in c['children']:
                    compartments[child_id]['parent'] = parent_id

        # filter out the orphans
        sparsified_compartments = { k:v for k,v in compartments.iteritems() if keep[k] }

        if compress_ids:
            ids = sorted(sparsified_compartments.keys(), key=lambda x: int(x))
            id_hash = { fid:str(i+1) for i,fid in enumerate(ids) }
            id_hash["-1"] = "-1"

            # build the final compartment index
            out_compartments = {}
            for cid, compartment in sparsified_compartments.iteritems():
                compartment['id'] = id_hash[cid]
                compartment['parent'] = id_hash[compartment['parent']]
                out_compartments[compartment['id']] = compartment

            return Morphology(compartment_index=out_compartments)
        else:
            return Morphology(compartment_index=sparsified_compartments)


def str_to_num(s):
    """ Try to convert a string s into a number """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
