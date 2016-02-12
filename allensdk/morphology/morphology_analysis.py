import math
from allensdk.core.swc import *
import sys
import numpy as np

VOID = 1000000000

def real(x):
    return x.real

def dist(a,b):
    dx = a[NODE_X] - b[NODE_X]
    dy = a[NODE_Y] - b[NODE_Y]
    dz = a[NODE_Z] - b[NODE_Z]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def angle(a,b,c):
    dist_ab = dist(a, b)
    dist_ac = dist(a, c)
    if dist_ab == 0 or dist_ac == 0:        
        print ("Warning. Parent and child SWC nodes have same coordinates. No bifurcation angle to compute.")
        print ("Parent at  %f,%f,%f" % (a[NODE_X], a[NODE_Y], a[NODE_Z]))
        print ("Child 0 at %f,%f,%f" % (b[NODE_X], b[NODE_Y], b[NODE_Z]))
        print ("Child 1 at %f,%f,%f" % (c[NODE_X], c[NODE_Y], c[NODE_Z]))
        return float('nan')
    return(math.acos((((b)[NODE_X]-(a)[NODE_X])*((c)[NODE_X]-(a)[NODE_X])+((b)[NODE_Y]-(a)[NODE_Y])*((c)[NODE_Y]-(a)[NODE_Y])+((b)[NODE_Z]-(a)[NODE_Z])*((c)[NODE_Z]-(a)[NODE_Z]))/(dist_ab*dist_ac))*180.0/math.pi)

# move forward and find indices (IDs) of furthest nodes along each
#   branch that are leaf nodes or that are right before the next
#   bifurcation
def get_remote_child(nt, t):
    rchildlist = []
    root_seg = nt.node(t)
    children = root_seg[NODE_CHILDREN]
    for seg in children:
        tmp = seg[NODE_ID]
        child_seg = seg
        while len(child_seg[NODE_CHILDREN]) == 1:
            child_seg = child_seg[NODE_CHILDREN][0]
            tmp = child_seg[NODE_ID]
        rchildlist.append(tmp)
    return rchildlist


class MorphologyFeatureExtractor( object ):
    def __init__(self):
        self.Width=0.0
        self.Height=0.0
        self.Depth=0.0
        self.Diameter=0.0
        self.Length=0.0
        self.Volume=0.0
        self.Surface=0.0
        self.Hausdorff=0.0

        self.N_node=0
        self.N_stem=0
        self.N_bifs=0
        self.N_branch=0
        self.N_tips=0
        self.Max_Order=0

        self.Pd_ratio=0.0
        self.Contraction=0.0
        self.Max_Eux=0.0
        self.Max_Path=0.0
        self.BifA_local=0.0
        self.BifA_remote=0.0
        self.Soma_surface=0.0
        self.Fragmentation=0.0


    def compute_features(self, nt):
        self.root_node = nt.soma
        self.N_node = nt.num_nodes
        self.N_stem = len(self.root_node[NODE_CHILDREN])
        self.Soma_surface = 4*math.pi*(self.root_node[NODE_R])*(self.root_node[NODE_R])

        self.compute_linear(nt)
        self.compute_tree(nt)

        features = np.zeros(22)
        features[0] = self.N_node        # feature # 0: Number of Nodes          OK
        features[1] = self.Soma_surface  # feature # 1: Soma Surface             OK
        features[2] = self.N_stem        # feature # 2: Number of Stems          OK
        features[3] = self.N_bifs        # feature # 3: Number of Bifurcations   OK
        features[4] = self.N_branch      # feature # 4: Number of Branches       OK
        features[5] = self.N_tips        # feature # 5: Number of Tips           OK
        features[6] = self.Width         # feature # 6: Overall Width            ?
        features[7] = self.Height        # feature # 7: Overall Height           ?
        features[8] = self.Depth         # feature # 8: Overall Depth            ?
        features[9] = self.Diameter      # feature # 9: Average Diameter         OK
        features[10] = self.Length       # feature # 10: Total Length            OK
        features[11] = self.Surface      # feature # 11: Total Surface           OK
        features[12] = self.Volume       # feature # 12: Total Volume            OK
        features[13] = self.Max_Eux      # feature # 13: Max Euclidean Distance  OK
        features[14] = self.Max_Path     # feature # 14: Max Path Distance       OK
        features[15] = self.Max_Order    # feature # 15: Max Branch Order        OK
        features[16] = self.Contraction  # feature # 16: Average Contraction     OK
        features[17] = self.Fragmentation    #feature # 17: Average Fragmentation
        features[18] = self.Pd_ratio     # feature # 18: Average Parent-daughter Ratio
        features[19] = self.BifA_local   # feature # 19: Average Bifurcation Angle Local
        features[20] = self.BifA_remote  # feature # 20: Average Bifurcation Angle Remote

        if self.N_branch == 0:
             features[21] = float('nan')
        else:
             features[21] = 1.0 * self.N_node / self.N_branch

        feature_desc = [
            "number_nodes",
            "soma_surface",
            "number_stems",
            "number_bifurcations",
            "number_branches", 
            "number_tips",
            "overall_width",
            "overall_height",
            "overall_depth",
            "average_diameter",
            "total_length",
            "total_surface",
            "total_volume",
            "max_euclidean_distance", 
            "max_path_distance",
            "max_branch_order", 
            "average_contraction",
            "average_fragmentation",
            "average_parent_daughter_ratio", 
            "average_bifurcation_angle_local", 
            "average_bifurcation_angle_remote", 
            "nodes_over_branches"
            ]

        return features, feature_desc, { feature_desc[i]:features[i] for i in range(len(feature_desc)) }



    # do a search along the list to compute overall N_bif, 
    #   N_tip, width, height, depth, length, volume, surface, 
    #   average diameter and max euclidean distance.
    def compute_linear(self, nt):
        xmin = VOID
        ymin = VOID
        zmin = VOID
        xmax = 0.0
        ymax = 0.0
        zmax = 0.0

        for curr in nt.compartment_list:
            xmin = min(xmin,curr[NODE_X])
            ymin = min(ymin,curr[NODE_Y])
            zmin = min(zmin,curr[NODE_Z])
            xmax = max(xmax,curr[NODE_X])
            ymax = max(ymax,curr[NODE_Y])
            zmax = max(zmax,curr[NODE_Z])
            if len(curr[NODE_CHILDREN]) == 0:
                self.N_tips += 1
            elif len(curr[NODE_CHILDREN]) > 1:
                self.N_bifs += 1
            parent = nt.parent_of(curr)
            if parent is None:
                continue
            l = dist(curr,parent);
            self.Diameter += 2*curr[NODE_R];
            self.Length += l;
            self.Surface += 2*math.pi*curr[NODE_R]*l
            self.Volume += math.pi*curr[NODE_R]*curr[NODE_R]*l
            lsoma = dist(curr, self.root_node)
            self.Max_Eux = max(self.Max_Eux,lsoma)
        self.Width = xmax-xmin
        self.Height = ymax-ymin
        self.Depth = zmax-zmin
        self.Diameter /= nt.num_nodes


    # do a search along the tree to compute N_branch, max path distance, 
    #   max branch order, average Pd_ratio, average Contraction, 
    #   average Fragmentation, average bif angle local & remote
    def compute_tree(self, nt):
        pathTotal = np.zeros(nt.num_nodes)
        depth = np.zeros(nt.num_nodes)

        stack = []
        stack.append(self.root_node[NODE_ID])
        pathlength = 0.0
        eudist = 0.0
        max_local_ang = 0.0
        max_remote_ang = 0.0

        N_ratio = 0
        N_Contraction = 0

        if len(self.root_node[NODE_CHILDREN]) > 1:
            local_ang = 0.0
            remote_ang = 0.0
            max_local_ang = 0.0
            max_remote_ang = 0.0
            ch_local1 = self.root_node[NODE_CHILDREN][0][NODE_ID]
            ch_local2 = self.root_node[NODE_CHILDREN][1][NODE_ID]
            local_ang = angle(self.root_node,nt.node(ch_local1),nt.node(ch_local2))

            remotes = get_remote_child(nt, self.root_node)
            ch_remote1 = remotes[0]
            ch_remote2 = remotes[1]
            remote_ang = angle(self.root_node, nt.node(ch_remote1), nt.node(ch_remote2));
            if local_ang == local_ang:
                max_local_ang = max(max_local_ang,local_ang);
            if remote_ang == remote_ang:
                max_remote_ang = max(max_remote_ang,remote_ang);

            self.BifA_local += max_local_ang
            self.BifA_remote += max_remote_ang

        t = None
        tmp = None
        fragment = None
        while len(stack) > 0:
            t = stack.pop()
            for child in nt.node(t)[NODE_CHILDREN]:
                self.N_branch += 1
                if nt.node(t)[NODE_R] > 0:
                    N_ratio += 1
                    self.Pd_ratio += 1.0*child[NODE_R]/nt.node(t)[NODE_R]
                pathlength = dist(child, nt.node(t))

                fragment = 0.0
                child_id = child[NODE_ID]
                while len(nt.node(child_id)[NODE_CHILDREN]) == 1:
                    ch = nt.node(child_id)[NODE_CHILDREN][0][NODE_ID]
                    pathlength += dist(nt.node(ch), nt.node(child_id))
                    fragment += 1
                    child_id = ch
                eudist = dist(nt.node(child_id), nt.node(t))
                self.Fragmentation += fragment
                if pathlength > 0:
                    self.Contraction += eudist/pathlength
                    N_Contraction += 1

                #we are reaching a tip point or another branch point, 
                #   computation for this branch is over
                chsz = len(nt.node(child_id)[NODE_CHILDREN])
                if chsz > 1:  #another branch
                    stack.append(child_id)

                    #compute local bif angle and remote bif angle
                    local_ang = 0.0
                    remote_ang = 0.0
                    max_local_ang = 0
                    max_remote_ang = 0
                    ch_local1 = nt.node(child_id)[NODE_CHILDREN][0][NODE_ID]
                    ch_local2 = nt.node(child_id)[NODE_CHILDREN][1][NODE_ID]
                    local_ang = angle(nt.node(child_id),nt.node(ch_local1),nt.node(ch_local2))

                    remotes = get_remote_child(nt, child_id)
                    ch_remote1 = remotes[0]
                    ch_remote2 = remotes[1]
                    remote_ang = angle(nt.node(child_id),nt.node(ch_remote1),nt.node(ch_remote2));
                    if local_ang == local_ang:
                        max_local_ang = max(max_local_ang,local_ang)
                    if remote_ang == remote_ang:
                        max_remote_ang = max(max_remote_ang,remote_ang)
                    self.BifA_local += max_local_ang
                    self.BifA_remote += max_remote_ang

                pathTotal[child_id] = pathTotal[t] + pathlength
                depth[child_id] = depth[t] + 1
        if N_ratio == 0:
            self.Pd_ratio = float('nan')
        else:
            self.Pd_ratio /= N_ratio

        if  self.N_branch == 0 :
            self.Fragmentation = float('nan')
        else:
            self.Fragmentation /= self.N_branch

        if N_Contraction == 0:
            self.Contraction = float('nan')
        else:
            self.Contraction /= N_Contraction

        if self.N_bifs==0:
            self.BifA_local = 0
            self.BifA_remote = 0
        else:
            self.BifA_local /= self.N_bifs
            self.BifA_remote /= self.N_bifs

        for i in range(nt.num_nodes):
            self.Max_Path = max(self.Max_Path, pathTotal[i])
            self.Max_Order = max(self.Max_Order, depth[i])

