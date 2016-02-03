#!/usr/bin/python
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

import csv
import copy
import math

########################################################################
# morphology nodes
#
# morphology nodenodes are stored as dicts and have the following fields
# the first seven correspond directly to fields in SWC files. SWC fields
#   are numeric
NODE_ID      = 'id'
NODE_TYPE    = 'type'
NODE_X       = 'x'
NODE_Y       = 'y'
NODE_Z       = 'z'
NODE_R       = 'radius'
NODE_PN      = 'parent'
SWC_COLUMNS = [ NODE_ID, NODE_TYPE, NODE_X, NODE_Y, NODE_Z, NODE_R, NODE_PN ]
# additional node data
# each unconnected graph has its own ID. this is the ID of graph that the
#   node resides in
NODE_TREE_ID = 'tree_id'     
# a list references to child nodes
NODE_CHILDREN = 'children'   
# each object is tagged with a label to detect type errors
RTTI = 'rtti'   # type information label
MORPHOLOGY_NODE   = "morphology node" # identifying node tag

def print_node(seg):
    # verify passed value is morphology node
    if RTTI not in seg or seg[RTTI] != MORPHOLOGY_NODE:
        raise TypeError("Object not recognized as morphology node")
    disp = "SWC node: "
    disp += "%d " % seg[NODE_ID]
    disp += "%d " % seg[NODE_TYPE]
    disp += "%f " % seg[NODE_X]
    disp += "%f " % seg[NODE_Y]
    disp += "%f " % seg[NODE_Z]
    disp += "%f " % seg[NODE_R]
    disp += "%d [" % seg[NODE_PN]
    ch = ""
    for i in range(len(seg[NODE_CHILDREN])):
        ch += "%d " % seg[NODE_CHILDREN][i][NODE_ID]
    if len(ch) > 0:
        ch = ch[:-1]
    disp += ch[:] + "] %d" % seg[NODE_TREE_ID]
    print disp


########################################################################
def read_swc(file_name, columns=None, numeric_columns="NOT_USED"):
    """  Read in an SWC file and return a Morphology object.
    SWC are basically CSV files, but they often don't have headers. 
    You can pass those in explicitly and also indicate which
    columns are numeric.

    Parameters
    ----------
    file_name: string
        SWC file name.

    columns: list of strings
        names of the columns in this file (default: SWC_COLUMNS)

    Returns
    -------
    Morphology
        A Morphology instance.
    """
    if columns == None:
        columns = SWC_COLUMNS

    # open file and 
    rows = open(file_name, "r").readlines()

    # skip comment rows, strip off extra whitespace
    rows = [ r.strip() for r in rows if len(r) > 0 and r.strip()[0] != '#' ]

    # parse input 
    reader = csv.DictReader(rows, 
                    fieldnames=columns, 
                    delimiter=' ', 
                    skipinitialspace=True, 
                    restkey='other')

    # convert numeric columns and create compartment list
    compartment_list = []
    for compartment in reader:
        for k in compartment:
            compartment[k] = str_to_num(compartment[k])
        compartment_list.append(compartment)

    # return new Morphology object
    return Morphology(compartment_list=compartment_list)    


def read_rows(rows, columns, numeric_columns="NOT_USED"):
    raise AssertionError("This function is deprecated. It can be restored if it looks to be important")

def read_string(s, columns=SWC_COLUMNS, numeric_columns="NOT_USED"):
    raise AssertionError("This function is deprecated. It can be restored if it looks to be important")


########################################################################
########################################################################

class Morphology( object ):
    """ Keep track of the list of compartments in a morphology and provide 
    a few helper methods (index by id, sparsify, root, etc).  
    During initialization the compartments are assigned a 'children' 
    property that is a list of IDs to child compartments.
    """

    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4

    NODE_TYPES = [ SOMA, AXON, BASAL_DENDRITE, APICAL_DENDRITE ]

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

        ##############################################
        self._compartment_list = []
        self._compartment_index = {}
        # define tree list here for clarity, even though it's reset below
        #   when nodes are assigned
        self._tree_list = []
        
        ##############################################
        # construct the compartment list and index
        # first try to do so using the compartment list, then try using
        #   the compartment index and if that fails then complain
        if compartment_list:
            self.compartment_list = compartment_list
        elif compartment_index:
            self.compartment_index = compartment_index
        else:
            raise AssertionError("No compartments provided to Morhphology constructor")
        ##############################################
        # verify morphology is consistent with morphology rules (e.g.,
        #   no dendrite branching from an axon)
        num_errors = self.check_consistency()
        if num_errors > 0:
            raise ValueError("Morphology appears to be inconsistent")

        ##############################################
        # root node (this must be part of the soma)
        self._soma = None
        for i in range(len(self.compartment_list)):
            seg = self.compartment_list[i]
            if seg[NODE_TYPE] == Morphology.SOMA and seg[NODE_PN] < 0:
                if self._soma is not None:
                    raise ValueError("Multiple somas detected in SWC file")
                self._soma = seg
        # TODO
        #self.check_consistency()

    ########################

    @property 
    def compartment_list(self):
        """ Return the compartment list.  This is a property to ensure that the 
        compartment list and compartment index are in sync. """
        return self._compartment_list

    @compartment_list.setter
    def compartment_list(self, compartment_list):
        """ Update the compartment list.  Update the compartment index. """
        self._set_compartments(compartment_list)

    @property
    def compartment_index(self):
        """ Return the compartment index.  This is a property to ensure that the
        compartment list and compartment index are in sync. """
        return self._compartment_index

    @compartment_index.setter
    def compartment_index(self, compartment_index):
        """ Update the compartment index.  Update the compartment list. """
        self._set_compartments(compartment_index.values())

    @property
    def num_trees(self):
        return len(self._tree_list)

    # TODO add filter for number of nodes of a particular type
    @property
    def num_nodes(self):
        return len(self.compartment_list)

    # internal function
    # take a list of SWC-like objects and turn those into morphology
    #   nodes
    # need to be able to initialize from a list supplied by an SWC file
    #   while also being able to initialize from the compartment list of
    #   an existing Morphology object. As nodes in a morphology object
    #   contain reference to nodes in that object, make a shallow copy
    #   of input nodes and overwrite known references (ie, the 
    #   'children' array)
    def _set_compartments(self, compartment_list):
        self._compartment_list = []
        for obj in compartment_list:
            seg = copy.copy(obj)
            seg[NODE_TREE_ID] = -1
            seg[NODE_CHILDREN] = []
            seg[RTTI] = MORPHOLOGY_NODE
            self._compartment_list.append(seg)
        # list data now set. remove holes in sequence and re-index
        self.reconstruct()

    ####################################################################

    def tree(self, n):
        """ Returns a list of all nodes within the specified tree.
        A tree is defined as a fully connected graph of nodes. Each
        tree has exactly one root.

        Parameters
        ----------
        n: integer
            ID of desired tree
            
        Returns
        -------
        A list of all morphology objects in the specified tree, or None
        if the tree doesn't exist
        """
        if n < 0 or n >= len(self._tree_list):
            return None
        return self._tree_list[n]


    def node(self, n):
        """ Returns the morphology node having the specified ID.

        Parameters
        ----------
        n: integer
            ID of desired node
            
        Returns
        -------
        A morphology object having the specified ID, or None if it doesn't
        exist
        """
#        if n < 0 or n >= len(self._compartment_list):
#            return None
#        return self._compartment_list[n]
        return self.resolve_node_type(n)

    # returns a list of nodes located within dist of x,y,z
    def find(self, x, y, z, dist, node_type=None):
        found = []
        for seg in self.compartment_list:
            dx = seg[NODE_X] - x
            dy = seg[NODE_Y] - y
            dz = seg[NODE_Z] - z
            if math.sqrt(dx*dx + dy*dy + dz*dz) <= dist:
                if node_type is None or seg[NODE_TYPE] == node_type:
                    found.append(seg)
        return found


    def parent_of(self, seg):
        """ Returns parent of the specified node

        Parameters
        ----------
        seg: integer or Morphology Object
            The ID of the child node, or the child node itself
            
        Returns
        -------
        A morphology object, or None if no parent exists or if the
        specified node ID doesn't exist
        """
        # handle case when index passed
        if type(seg).__name__ == 'int':
            if seg < 0 or seg >= len(self._compartment_list):
                return None
            seg = self._compartment_list[seg]
        # handle case when node (dictionary) passed
        elif type(seg).__name__ == 'dict':
            if RTTI not in seg or seg[RTTI] != MORPHOLOGY_NODE:
                raise TypeError("Object not recognized as morphology node")
        # no luck. try converting it to an int
        else:
            try:
                seg = int(seg)
                if seg < 0 or seg >= len(self._compartment_list):
                    return None
                seg = self._compartment_list[seg]
            except ValueError:
                raise TypeError("Object not recognized as morphology node or index")
        if seg[NODE_PN] >= 0:
            return self._compartment_list[seg[NODE_PN]]
        return None

    def resolve_node_type(self, seg):
        if type(seg).__name__ == 'int':
            if seg < 0 or seg >= len(self._compartment_list):
                raise ValueError("Specified child (%d) is not a valid ID" % seg)
            return self._compartment_list[seg]
        elif type(seg).__name__ == 'dict':
            if RTTI not in seg or seg[RTTI] != MORPHOLOGY_NODE:
                raise TypeError("Object not recognized as morphology node")
        # no luck. try converting it to an int
        else:
            try:
                seg = int(seg)
                if seg < 0 or seg >= len(self._compartment_list):
                    return None
                seg = self._compartment_list[seg]
            except ValueError:
                raise TypeError("Object not recognized as morphology node or index")
        return seg

    def change_parent(self, child, parent):
        child_seg = self.resolve_node_type(child)
        parent_seg = self.resolve_node_type(parent)
        # if child has former parent, remove it from parent's child list
        if child_seg[NODE_PN] >= 0:
            old_par = self.node(child_seg[NODE_PN])
            old_par[NODE_CHILDREN].remove(child_seg)
        parent_seg[NODE_CHILDREN].append(child_seg)
        child_seg[NODE_PN] = parent_seg[NODE_ID]
            

    def children(self, seg):
        """ Returns a list of the children of the specified node

        Parameters
        ----------
        seg: integer or Morphology Object
            The ID of the parent node, or the parent node itself
            
        Returns
        -------
        A list of the child morphology objects. If the ID of the parent
        node is invalid, None is returned.
        """
        seg = self.resolve_node_type(seg)
#        # handle case when index passed
#        if type(seg).__name__ == 'int':
#            if seg < 0 or seg >= len(self._compartment_list):
#                return None
#            seg = self._compartment_list[seg]
#        # handle case when node (dictionary) passed
#        elif type(seg).__name__ == 'dict':
#            if RTTI not in seg or seg[RTTI] != MORPHOLOGY_NODE:
#                raise TypeError("Object not recognized as morphology node")
#        # no luck. try converting it to an int
#        else:
#            try:
#                seg = int(seg)
#                if seg < 0 or seg >= len(self._compartment_list):
#                    return None
#                seg = self._compartment_list[seg]
#            except ValueError:
#                raise TypeError("Object not recognized as morphology node or index")
        return seg[NODE_CHILDREN]

    @property
    def soma(self):
        """ [deprecated] Returns root node of soma, if present"""
        return self._soma

    @property
    def root(self):
        """ [deprecated] Returns root node of soma, if present. Use 'soma' instead of 'root'"""
        return self._soma

    def compartment_index_by_type(self, compartment_type):
        """ Return an dictionary of compartments indexed by id that all have
        a particular compartment type.

        Parameters
        ----------
        compartment_type: int
            Desired compartment type
        """

        return { c[NODE_ID]: c for c in self._compartment_list if c[NODE_TYPE] == compartment_type }


    def save(self, file_name):
        """ Write this morphology out to an SWC file 
      
        Parameters
        ----------
        file_name: string
            desired name of your SWC file

        """
        f = open(file_name, "w")
        f.write("#n,type,x,y,z,radius,parent\n")
        for seg in self.compartment_list:
            f.write("%d %d " % (seg[NODE_ID], seg[NODE_TYPE]))
            f.write("%0.4f " % seg[NODE_X])
            f.write("%0.4f " % seg[NODE_Y])
            f.write("%0.4f " % seg[NODE_Z])
            f.write("%0.4f " % seg[NODE_R])
            f.write("%d\n" % seg[NODE_PN])
        f.close()


    # keep for backward compatibility, but don't publish in docs
    def write(self, file_name):
        self.save(file_name)


    def sparsify(self, modulo, compress_ids=False):
        """ Return a new Morphology object that has a given number of non-leaf,
        non-root nodes removed.  IDs can be reassigned so as to be continuous.

        Parameters
        ----------
        modulo: int
           keep 1 out of every modulo nodes.

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
            pid = c[NODE_PN]
            cid = c[NODE_ID]
            ctype = c[NODE_TYPE]

            # keep the root, soma, junctions, and the first child of the root (for visualization)
            #if pid == "-1" or len(c[NODE_CHILDREN]) != 1 or pid == root[NODE_ID] or ctype == Morphology.SOMA:
            if pid < 0 or len(c[NODE_CHILDREN]) != 1 or pid == root[NODE_ID] or ctype == Morphology.SOMA:
                keep[cid] = True
            else:
                keep[cid] = (ct % modulo) == 0
                
            ct += 1

        # hook children up to their new parents
        for i, c in compartments.iteritems():
            comp_id = c[NODE_ID]

            if keep[comp_id] is False:
                parent_id = c[NODE_PN]
                while keep[parent_id] is False:
                    parent_id = compartments[parent_id][NODE_PN]

                for child_id in c[NODE_CHILDREN]:
                    compartments[child_id][NODE_PN] = parent_id

        # filter out the orphans
        sparsified_compartments = { k:v for k,v in compartments.iteritems() if keep[k] }

        if compress_ids:
            ids = sorted(sparsified_compartments.keys(), key=lambda x: int(x))
            id_hash = { fid:str(i+1) for i,fid in enumerate(ids) }
            id_hash[-1] = -1

            # build the final compartment index
            out_compartments = {}
            for cid, compartment in sparsified_compartments.iteritems():
                compartment[NODE_ID] = id_hash[cid]
                compartment[NODE_PN] = id_hash[compartment[NODE_PN]]
                out_compartments[compartment[NODE_ID]] = compartment

            return Morphology(compartment_index=out_compartments)
        else:
            return Morphology(compartment_index=sparsified_compartments)

    ####################################################################
    ####################################################################
    #
    # internal function that re-orders data, removing 'holes' in sequence
    #   so that each object ID corresponds to its position in list
    # parent-child indices are recalculated as is compartment table
    def reconstruct(self):
        # construct a map between new and old IDs
        remap = {}
        # everything defaults to root. this way if a parent was deleted
        #   the child will become a new root
        for i in range(len(self.compartment_list)):
            remap[i] = -1
        # map old old node numbers to new ones. reset n to the new ID
        #   and put node in new list
        new_id = 0
        tmp_list = []
        for seg in self.compartment_list:
            if seg is not None:
                remap[seg[NODE_ID]] = new_id
                seg[NODE_ID] = new_id
                tmp_list.append(seg)
                new_id += 1
        # use map to reset parent values. copy objs to new list
        for seg in tmp_list:
            if seg[NODE_PN] >= 0:
                seg[NODE_PN] = remap[seg[NODE_PN]]
        # replace compartment list with newly created node list
        self._compartment_list = tmp_list
        # reconstruct parent/child relationship links
        # forget old relations
        for seg in self.compartment_list:
            seg[NODE_CHILDREN] = []
        # add each object to its parents child list
        for seg in self.compartment_list:
            par_num = seg[NODE_PN]
            if par_num >= 0:
                self.compartment_list[par_num][NODE_CHILDREN].append(seg)
        # update tree lists
        self.separate_trees()
        ############################
        # Rebuild internal index and links between parents and children
        self._compartment_index = { c[NODE_ID]: c for c in self.compartment_list }
        # compartment list is complete and sequential so don't need index
        #   to resolve relationships
        # for each node, reset children array
        # for each node, add self to parent's child list
        for seg in self._compartment_list:
            seg[NODE_CHILDREN] = []
        for seg in self._compartment_list:
            if seg[NODE_PN] >= 0:
                self._compartment_list[seg[NODE_PN]][NODE_CHILDREN].append(seg)
        # verify that each node ID is the same as its position in the
        #   compartment list
        for i in range(len(self.compartment_list)):
            if i != self.node(i)[NODE_ID]:
                raise Assertion_Error("Internal error detected -- compartment list not properly formed")


    def append(self, node_list):
        """ Add additional nodes to this Morphology. Those nodes must
        originate from another morphology object.

        Parameters
        ----------
        node_list: list of Morphology nodes
        """
        # construct a map between new and old IDs of added nodes
        remap = {}
        for i in range(len(node_list)):
            remap[i] = -1
        # map old old node numbers to new ones. reset n to the new ID
        # append new nodes to existing node list
        old_count = len(self.compartment_list)
        new_id = old_count
        for seg in node_list:
            if seg is not None:
                remap[seg[NODE_ID]] = new_id
                seg[NODE_ID] = new_id
                self._compartment_list.append(seg)
                new_id += 1
        # use map to reset parent values. copy objs to new list
        for i in range(old_count, len(self.compartment_list)):
            seg = self.compartment_list[i]
            if seg[NODE_PN] >= 0:
                seg[NODE_PN] = remap[seg[NODE_PN]]
        self.reconstruct()


    def stumpify_axon(self, count=10):
        """ remove all axon compartments except the first 'count' nodes
        of the connected axon root
        """
        # find connected axon root
        axon_root = None
        for seg in self.compartment_list:
            if seg[NODE_TYPE] == Morphology.AXON:
                par_id = seg[NODE_PN]
                if par_id >= 0:
                    par = self.compartment_list[par_id]
                    if par[NODE_TYPE] != Morphology.AXON:
                        axon_root = seg
                        break
        if axon_root is None:
            return
        # flag the first 'count' nodes from the axon root
        ax = axon_root
        for i in range(count):
            # ignore bifurcations -- go 'count' deep on one line only
            ax["flag"] = i
            children = ax[NODE_CHILDREN]
            if len(children) > 0:
                ax = children[0]
        # strip out all axons that aren't flagged
        for i in range(len(self.compartment_list)):
            seg = self.compartment_list[i]
            if seg[NODE_TYPE] == Morphology.AXON:
                if "flag" not in seg:
                    self.compartment_list[i] = None
        self.reconstruct()
            
        
    # strip out everything but the soma and the specified SWC type
    def strip_all_other_types(self, obj_type, keep_soma=True):
        for i in range(len(self.compartment_list)):
            seg = self.compartment_list[i]
            if seg[NODE_TYPE] == obj_type or (keep_soma and seg[NODE_TYPE] == 1):
                continue
            # make children forget about removed parents
            for child in seg[NODE_CHILDREN]:
                child[NODE_PN] = -1
            # remove node from list
            self.compartment_list[i] = None
        self.reconstruct()
    
    # strip out the specified SWC type
    def strip_type(self, obj_type):
        for i in range(len(self.compartment_list)):
            seg = self.compartment_list[i]
            if seg[NODE_TYPE] == obj_type:
                # make children forget about removed parents
                for child in seg[NODE_CHILDREN]:
                    child[NODE_PN] = -1
                # remove node from list
                self.compartment_list[i] = None
            else:
                # make sure parent isn't being removed. if so, unlink
                par_num = seg[NODE_PN]
                if par_num >= 0:
                    par_type = self.compartment_list[par_num][NODE_TYPE]
                    if par_type == obj_type:
                        seg[NODE_PN] = -1
        self.reconstruct()
    

    def apply_affine(self, aff, scale=None):
        """ Apply an affine transform to all compartments in this 
        morphology. Node radius is adjusted as well.
        
        Format of the affine matrix is:

        [x0 y0 z0 tx]
        [x1 y1 z1 ty]
        [x2 y2 z2 tz]

        where the left 3x3 portion of the matrix defines the affine
        rotation and scaling, and the right column is the translation
        vector

        The matrix must be collapsed and stored as a list as follows:

        [x0 y0, z0, x1, y1, z1, x2, y2, z2, tx, ty, tz]
        
        Parameters
        ----------
        aff: 3x4 array of floats (python 2D list, or numpy 2D array)
            the transformation matrix
        """
        # In addition to transforming the locations of the morphology
        #   nodes, the radius of each node must be adjusted.
        # There are 2 ways to measure scale from a transform. Assuming
        #   an isotropic transform, the scale is the cube root of the
        #   matrix determinant. The other ways is to measure scale 
        #   independently along each axis.
        # For now, the node radius is only updated based on the average
        #   scale along all 3 axes (eg, isotropic assumption), so calculate
        #   scale using the determinant
        #
        if scale is None:
            # calculate the determinant
            det0 = aff[0] * (aff[4]*aff[8] - aff[5]*aff[7])
            det1 = aff[1] * (aff[3]*aff[8] - aff[5]*aff[6])
            det2 = aff[2] * (aff[3]*aff[7] - aff[4]*aff[6])
            det = det0 + det1 + det2
            # determinant is change of volume that occurred during transform
            # assume equal scaling along all axes. take 3rd root to get
            #   scale factor
            det_scale = math.pow(abs(det), 1.0/3.0)
            ## measure scale along each axis
            ## keep this code here in case 
            #scale_x = abs(aff[0] + aff[3] + aff[6])
            #scale_y = abs(aff[1] + aff[4] + aff[7])
            #scale_z = abs(aff[2] + aff[5] + aff[8])
            #avg_scale = (scale_x + scale_y + scale_z) / 3.0;
            #
            # use determinant for scaling for now as it's most simple
            scale = det_scale
        for seg in self.compartment_list:
            x = seg[NODE_X]*aff[0] + seg[NODE_Y]*aff[1] + seg[NODE_Z]*aff[2] + aff[9]
            y = seg[NODE_X]*aff[3] + seg[NODE_Y]*aff[4] + seg[NODE_Z]*aff[5] + aff[10]
            z = seg[NODE_X]*aff[6] + seg[NODE_Y]*aff[7] + seg[NODE_Z]*aff[8] + aff[11]
            seg[NODE_X] = x
            seg[NODE_Y] = y
            seg[NODE_Z] = z
            seg[NODE_R] *= scale

    # construct list of independent trees (each tree has a root of -1)
    def separate_trees(self):
        trees = []
        # reset each node's tree ID to indicate that it's not assigned
        for seg in self.compartment_list:
            seg[NODE_TREE_ID] = -1
        # construct trees for each node
        # if a node is adjacent an existing tree, merge to it
        # if a node is adjacent multiple trees, merge all
        for seg in self.compartment_list:
            # see what trees this node is adjacent to
            local_trees = []
            if seg[NODE_PN] >= 0 and self.compartment_list[seg[NODE_PN]][NODE_TREE_ID] >= 0:
                local_trees.append(self.compartment_list[seg[NODE_PN]][NODE_TREE_ID])
            for child in seg[NODE_CHILDREN]:
                if child[NODE_TREE_ID] >= 0:
                    local_trees.append(child[NODE_TREE_ID])
            # figure out which tree to put node into
            # if there are muliple possibilities, merge all of them
            if len(local_trees) == 0:
                tree_num = len(trees) # create new tree
            elif len(local_trees) == 1:
                tree_num = local_trees[0]   # use existing tree
            elif len(local_trees) > 1:
                # this node is an intersection of multiple trees
                # merge all trees into the first one found
                tree_num = local_trees[0]
                for j in range(1,len(local_trees)):
                    dead_tree = local_trees[j]
                    trees[dead_tree] = []
                    for node in self.compartment_list:
                        if node[NODE_TREE_ID] == dead_tree:
                            node[NODE_TREE_ID] = tree_num
            # merge node into tree
            # ensure there's space
            while len(trees) <= tree_num:
                trees.append([])
            trees[tree_num].append(seg)
            seg[NODE_TREE_ID] = tree_num
        # consolidate tree lists into class's tree list object
        self._tree_list = []
        for tree in trees:
            if len(tree) > 0:
                self._tree_list.append(tree)
        # make soma's tree be the first tree, if soma present
        # this should be the case if the file is properly ordered, but
        #   don't assume that
        soma_tree = -1
        for seg in self.compartment_list:
            if seg[NODE_TYPE] == 1:
                soma_tree = seg[NODE_TREE_ID]
                break
        if soma_tree > 0:
            # swap soma tree for first tree in list
            tmp = self._tree_list[soma_tree]
            self._tree_list[soma_tree] = self._tree_list[0]
            self._tree_list[0] = tmp
        # reset node tree_id to correct tree number
        self.reset_tree_ids()


    def reset_tree_ids(self):
        # reset node tree_id to correct tree number
        for i in range(len(self._tree_list)):
            for j in range(len(self._tree_list[i])):
                self._tree_list[i][j][NODE_TREE_ID] = i

    # TODO verify that only recognized types are present
    # returns number of errors detected in file
    def check_consistency(self):
        # Make sure that the parents are of proper ID range
        n = self.num_nodes
        for seg in self.compartment_list:
            if seg[NODE_PN] >= 0:
                assert(seg[NODE_PN] < n)
        # make sure that each tree has exactly one root
        errs = 0
        for i in range(self.num_trees):
            tree = self.tree(i)
            root = -1
            for j in range(len(tree)):
                if tree[j][NODE_PN] == -1:
                    if root >= 0:
                        print("Too many roots in tree %d" % i)
                        errs += 1
                    root = j
            if root == -1:
                print("No root present in tree %d" % i)
                errs += 1
        # make sure each branch has at most one axon root
        # find type boundaries. at each axon boundary, walk back up
        #   tree to root and make sure another axon segment not
        #   encountered
        adoptees = self.find_type_boundary()
        for child in adoptees:
            if child[NODE_TYPE] == Morphology.AXON:
                par_id = child[NODE_PN]
                while par_id >= 0:
                    par = self.compartment_list[par_id]
                    if par[NODE_TYPE] == Morphology.AXON:
                        print("Branch has multiple axon roots")
                        print_node(child)
                        print_node(par)
                        errs += 1
                        break
                    par_id = par[NODE_PN]
        if errs > 0:
            print("Failed consistency check: %d errors encountered" % errs)
        return errs

    # return a list of segments who have parents that are a different type
    def find_type_boundary(self):
        adoptees = []
        for node in self.compartment_list:
            par = self.parent_of(node)
            if par is None:
                continue
            if node[NODE_TYPE] != par[NODE_TYPE]:
                adoptees.append(node)
        return adoptees

    # remove tree from swc's "forest"
    def delete_tree(self, n):
        if n < 0:
            return
        if n >= self.num_trees:
            print("Error -- attempted to delete non-existing tree (%d)" % n)
            raise ValueError
        tree = self.tree(n)
        for i in range(len(tree)):
            self.compartment_list[tree[i][NODE_ID]] = None
        del self._tree_list[n]
        self.reconstruct()
        # reset node tree_id to correct tree number
        self.reset_tree_ids()

    # code to assist in debugging
    def print_all_nodes(self):
        for node in self.compartment_list:
            print_node(node)


def str_to_num(s):
    """ Try to convert a string s into a number """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

########################################################################
########################################################################
# test code
#

err_cnt = 0

def reset_err_cnt():
    global err_cnt
    err_cnt = 0

def verify(expr, err_string):
    global err_cnt
    if not expr:
        print("Error: " + err_string)
        err_cnt += 1

def num_errors():
    global err_cnt
    return err_cnt

########################################
def test_1():
    # reduce list and identify 2 trees
    err = False
    fname = "a.swc"
    f = open(fname, "w")
    f.write("0 1 9 0 9 0.1 -1\n")
    f.write("1 2 9 1 9 0.1 0\n")
    f.write("3 2 9 2 9 0.1 1\n")
    f.write("4 2 9 3 9 0.1 -1\n")
    f.write("8 2 9 4 9 0.1 4\n")
    f.write("9 2 9 5 9 0.1 1\n")
    f.close()
    nrn = read_swc(fname)
    reset_err_cnt()
    #
    verify((nrn.num_nodes == 6), "incorrect number of nodes")
    for seg in nrn.compartment_list:
        verify((seg is not None), "NULL node found")
        verify((seg[NODE_Y] == seg[NODE_ID]), "Remmaping error (%d != %d)" % (seg[NODE_ID], seg[NODE_Y]))
    #
    verify((nrn.num_trees == 2), "incorrect number of trees")
    verify((len(nrn.tree(0)) == 4), "incorrect tree length (0)")
    verify((len(nrn.tree(1)) == 2), "incorrect tree length (1)")
    #
    verify((nrn.check_consistency() == 0), "consistency errors")
    #
    if num_errors() > 0:
        print("Failed test 1")
        nrn.print_all_nodes()
    else:
        print("Passed test #1")
    
########################################
def test_2():
    # similar to test 1, but with soma relocated to secondary tree
    fname = "b.swc"
    f = open(fname, "w")
    f.write("0 2 9 1 9 0.1 4\n")
    f.write("3 2 9 2 9 0.1 0\n")
    f.write("4 2 9 3 9 0.1 -1\n")
    f.write("8 2 9 4 9 0.1 4\n")
    f.write("9 2 9 5 9 0.1 10\n")
    f.write("10 1 9 0 9 0.1 -1\n")
    f.close()
    nrn = read_swc(fname)
    reset_err_cnt()
    verify((nrn.num_nodes == 6), "incorrect number of nodes")
    #
    verify((nrn.num_trees == 2), "incorrect number of trees")
    verify((len(nrn.tree(0)) == 2), "incorrect tree length (0)")
    verify((len(nrn.tree(1)) == 4), "incorrect tree length (1)")
    #
    verify((nrn.soma[NODE_TREE_ID] == 0), "Soma in wrong tree")
    #
    verify((nrn.check_consistency() == 0), "consistency errors")
    #
    if num_errors() > 0:
        print("Failed test 2")
        nrn.print_all_nodes()
    else:
        print("Passed test #2")


########################################
def test_3():
    # remove tree from 'forest'
    fname = "a.swc"
    f = open(fname, "w")
    f.write("0 1 9 0 9 0.1 -1\n")
    f.write("1 2 9 1 9 0.1 0\n")
    f.write("3 2 9 2 9 0.1 1\n")
    f.write("4 2 9 3 9 0.1 -1\n")
    f.write("8 2 9 4 9 0.1 4\n")
    f.write("9 2 9 5 9 0.1 1\n")
    f.close()
    nrn = read_swc(fname)
    reset_err_cnt()
    #
    verify((nrn.num_nodes == 6), "incorrect number of nodes")
    for seg in nrn.compartment_list:
        verify((seg[NODE_ID] == seg[NODE_Y]), "Segment %d has wrong ID" % seg[NODE_ID])
    #
    verify((nrn.num_trees == 2), "incorrect number of trees")
    nrn.delete_tree(1)
    verify((nrn.num_trees == 1), "failed ot delete tree")
    verify((len(nrn.tree(0)) == 4), "incorrect tree length (0)")
    #
    verify((nrn.check_consistency() == 0), "consistency errors")
    #
    if num_errors() > 0:
        print("Failed test 3")
        nrn.print_all_nodes()
    else:
        print("Passed test #3")
    
    
if __name__ == "__main__":
    test_1()
    test_2()
    test_3()

