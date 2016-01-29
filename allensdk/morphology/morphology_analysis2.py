import math
from swc import *
import sys
import numpy as np

########################################################################
########################################################################
# GMI calculations

VOID = 1000000000

def computeGMI(nt):
    # v3d version calculates 14 moments
    gmi = np.zeros(14)
    center_pos = np.zeros(3)
    avgR = 0.0;

    b = []
    for i in range(nt.num_nodes):
        b.append([])
        b[i] = np.zeros(4)
        b[i][0] = nt.node(i)[NODE_X]
        b[i][1] = nt.node(i)[NODE_Y]
        b[i][2] = nt.node(i)[NODE_Z]
        avgR += nt.node(i)[NODE_R]
        if nt.node(i)[NODE_PN] < 0:
            b[i][3] = -1;
        else:
            b[i][3] = nt.node(i)[NODE_PN]

    avgR /= nt.num_nodes
    gmi[13] = avgR

    m000 = compute_moments(b, nt.num_nodes, 0, 0, 0, VOID);
    center_pos[0] = compute_moments(b, nt.num_nodes, 1, 0, 0, VOID);
    center_pos[1] = compute_moments(b, nt.num_nodes, 0, 1, 0, VOID);
    center_pos[2] = compute_moments(b, nt.num_nodes, 0, 0, 1, VOID);

    for j in range(3):
        center_pos[j] /= m000
    compute_neuron_GMI(b, nt.num_nodes, center_pos, VOID, gmi);

    gmi_desc = []
    for i in range(len(gmi)):
        gmi_desc.append("moment%d" % (i+1))
    return gmi, gmi_desc

def real(x):
    return x.real

def compute_neuron_GMI(b_array, siz, center_pos, rad_thresh, gmi):
    if center_pos[0] != 0 or center_pos[1] != 0 or center_pos[2] != 0:
        for i in range(siz):
            for j in range(3):
                b_array[i][j] -= center_pos[j]

    c000 = compute_moments(b_array, siz, 0,0,0, rad_thresh)

    c200 = compute_moments(b_array, siz, 2,0,0, rad_thresh)
    c020 = compute_moments(b_array, siz, 0,2,0, rad_thresh)
    c002 = compute_moments(b_array, siz, 0,0,2, rad_thresh)
    c110 = compute_moments(b_array, siz, 1,1,0, rad_thresh)
    c101 = compute_moments(b_array, siz, 1,0,1, rad_thresh)
    c011 = compute_moments(b_array, siz, 0,1,1, rad_thresh)

    c300 = compute_moments(b_array, siz, 3,0,0, rad_thresh)
    c030 = compute_moments(b_array, siz, 0,3,0, rad_thresh)
    c003 = compute_moments(b_array, siz, 0,0,3, rad_thresh)
    c120 = compute_moments(b_array, siz, 1,2,0, rad_thresh)
    c102 = compute_moments(b_array, siz, 1,0,2, rad_thresh)
    c210 = compute_moments(b_array, siz, 2,1,0, rad_thresh)
    c201 = compute_moments(b_array, siz, 2,0,1, rad_thresh)
    c012 = compute_moments(b_array, siz, 0,1,2, rad_thresh)
    c021 = compute_moments(b_array, siz, 0,2,1, rad_thresh)
    c111 = compute_moments(b_array, siz, 1,1,1, rad_thresh)

    gmi[0] = c000
    gmi[1] = c200+c020+c002
    gmi[2] = c200*c020+c020*c002+c002*c200-c101*c101-c011*c011-c110*c110
    gmi[3] = c200*c020*c002-c002*c110*c110+2*c110*c101*c011-c020*c101*c101-c200*c011*c011

    spi = math.sqrt(math.pi)


    v_0_0  = complex((2*spi/3)*(c200+c020+c002),0)

    ####################################################################
    v_2_2 = complex(c200-c020,2*c110)
    v_2_2 *= spi*math.sqrt(2.0/15)

    v_2_1 = complex(-2*c101,-2*c011)
    v_2_1 *= spi*math.sqrt(2.0/15)

    v_2_0 = complex(2*c002-c200-c020,0)
    v_2_0 *= spi*math.sqrt(4.0/45)

    v_2_m1 = complex(2*c101,-2*c011)
    v_2_m1 *= spi*math.sqrt(2.0/15)

    v_2_m2 = complex(c200-c020,-2*c110)
    v_2_m2 *= spi*math.sqrt(2.0/15)

    ####################################################################
    v_3_3 = complex((-c300+3*c120) , (c030-3*c210))
    v_3_3 *= spi*math.sqrt(1.0/35)

    v_3_2 = complex((c201-c021), 2*c111)
    v_3_2 *= spi*math.sqrt(6.0/35)

    v_3_1 = complex((c300+c120-4*c102), (c030+c210-4*c012))
    v_3_1 *= spi*math.sqrt(3.0/175)

    v_3_0 = complex(2*c003 - 3*c201 - 3*c021,0)
    v_3_0 *= spi*math.sqrt(4.0/175)

    v_3_m1 = complex((-c300-c120+4*c102) , (c030+c210-4*c012))
    v_3_m1 *=  spi*math.sqrt(3.0/175)

    v_3_m2 = complex((c201-c021) , -2*c111)
    v_3_m2 *=  spi*math.sqrt(6.0/35)

    v_3_m3 = complex((c300-3*c120) , (c030-3*c210))
    v_3_m3 *= spi*math.sqrt(1.0/35)

    ####################################################################
    v_1_1 = complex((-c300-c120-c102), -(c030+c210+c012))
    v_1_1 *= spi*math.sqrt(6.0/25)

    v_1_0 = complex(c003+c201+c021,0)
    v_1_0 *= spi*math.sqrt(12.0/25)
    
    v_1_m1 = complex((c300+c120+c102) , -(c030+c210+c012))
    v_1_m1 *= spi*math.sqrt(6.0/25)

    ####################################################################
    v_g33_2_2 = math.sqrt(10.0/21)*v_3_3*v_3_m1 - math.sqrt(20.0/21)*v_3_2*v_3_0 + math.sqrt(2.0/7)*v_3_1*v_3_1
    v_g33_2_1 = math.sqrt(25.0/21)*v_3_3*v_3_m2 - math.sqrt(5.0/7)*v_3_2*v_3_m1 + math.sqrt(2.0/21)*v_3_1*v_3_0
    v_g33_2_0 = math.sqrt(25.0/21)*v_3_3*v_3_m3 - math.sqrt(3.0/7)*v_3_1*v_3_m1 + math.sqrt(4.0/21)*v_0_0*v_0_0
    v_g33_2_m1 = math.sqrt(25.0/21)*v_3_m3*v_3_2 - math.sqrt(5.0/7)*v_3_m2*v_3_1 + math.sqrt(2.0/21)*v_3_m1*v_3_0
    v_g33_2_m2 = math.sqrt(10.0/21)*v_3_m3*v_3_1 - math.sqrt(20.0/21)*v_3_m2*v_3_0 + math.sqrt(2.0/7)*v_3_m1*v_3_m1

    ####################################################################
    v_g31_2_2 = -math.sqrt(1.0/105)*v_3_2*v_1_0 + math.sqrt(1.0/35)*v_3_3*v_1_m1 + math.sqrt(1.0/525)*v_3_1*v_1_1
    v_g31_2_1 = math.sqrt(2.0/105)*v_3_2*v_1_m1 + math.sqrt(1.0/175)*v_3_0*v_1_1 - math.sqrt(4.0/525)*v_3_1*v_1_0
    v_g31_2_0 = -math.sqrt(3.0/175)*v_3_0*v_1_0 + math.sqrt(2.0/175)*v_3_1*v_1_m1 + math.sqrt(2.0/175)*v_3_m1*v_1_1
    v_g31_2_m1 = math.sqrt(2.0/105)*v_3_m2*v_1_1 + math.sqrt(1.0/175)*v_3_0*v_1_m1 -math.sqrt(4.0/525)*v_3_m1*v_1_0
    v_g31_2_m2 = -math.sqrt(1.0/105)*v_3_m2*v_1_0 + math.sqrt(1.0/35)*v_3_m3*v_1_1 + math.sqrt(1.0/525)*v_3_m1*v_1_m1

    ####################################################################
    v_g11_2_2 = 0.2*v_1_1*v_1_1
    v_g11_2_1 = math.sqrt(2.0/25)*v_1_0*v_1_1
    v_g11_2_0 = math.sqrt(2.0/75)*(v_1_0*v_1_0 + v_1_1*v_1_m1)
    v_g11_2_m1 = math.sqrt(2.0/25)*v_1_0*v_1_m1
    v_g11_2_m2 = 0.2*v_1_m1*v_1_m1

    ####################################################################
    tmp = v_0_0 ** (12.0/5)

    gmi[4] = ((1/math.sqrt(7.0)) * (v_3_3*v_3_m3*2.0 - v_3_2*v_3_m2*2.0 + v_3_1*v_3_m1*2.0 - v_3_0*v_3_0) / tmp).real
    gmi[5] = real((1/math.sqrt(3.0))* (v_1_1*v_1_m1*2.0 - v_1_0*v_1_0) / tmp)

    tmp = v_0_0 ** (24.0/5)
    #tmp = math.pow(v_0_0,(24.0/5))
    gmi[6] = real((1/math.sqrt(5.0)) * (v_g33_2_m2*v_g33_2_2*2.0 - v_g33_2_m1*v_g33_2_1*2.0 + v_g33_2_0*v_g33_2_0) / tmp)
    gmi[7] = real((1/math.sqrt(5.0)) * (v_g31_2_m2*v_g31_2_2*2.0 - v_g31_2_m1*v_g31_2_1*2.0 + v_g31_2_0*v_g31_2_0) / tmp)
    gmi[8] = real((1/math.sqrt(5.0)) * (v_g33_2_m2*v_g31_2_2 - v_g33_2_m1*v_g31_2_1 + v_g33_2_0*v_g31_2_0 - v_g33_2_1*v_g31_2_m1 + v_g33_2_2*v_g31_2_m2) / tmp)
    gmi[9] = real((1/math.sqrt(5.0)) * (v_g31_2_m2*v_g11_2_2 - v_g31_2_m1*v_g11_2_1 + v_g31_2_0*v_g11_2_0 - v_g31_2_1*v_g11_2_m1 + v_g31_2_1*v_g11_2_m2) / tmp)

    tmp = v_0_0 ** (17.0/5)
    gmi[10] = real((1/math.sqrt(5.0)) * (v_g33_2_m2*v_2_2 - v_g33_2_m2*v_2_1 + v_g33_2_0*v_2_0 - v_g33_2_1*v_2_m1 + v_g33_2_2*v_2_m2) / tmp)
    gmi[11] = real((1/math.sqrt(5.0)) * (v_g31_2_m2*v_2_2 - v_g31_2_m2*v_2_1 + v_g31_2_0*v_2_0 - v_g31_2_1*v_2_m1 + v_g31_2_2*v_2_m2) / tmp)
    gmi[12] = real((1/math.sqrt(5.0)) * (v_g11_2_m2*v_2_2 - v_g11_2_m2*v_2_1 + v_g11_2_0*v_2_0 - v_g11_2_1*v_2_m1 + v_g11_2_2*v_2_m2) / tmp)


########################################################################

def compute_moments(a_array, siz, p, q, r, rad_thres):
    m = 0.0
    step0=0.1;
    b1 = np.zeros(4)
    b2 = np.zeros(4)

    for i in range(siz):
        if a_array[i][3] < 0:
            continue
        total = 0.0
        for j in range(3):
            b1[j] = a_array[i][j]
            b2[j] = a_array[int(a_array[i][3])][j]
            total += (b1[j]-b2[j])*(b1[j]-b2[j])

        length = math.sqrt(total)
        K = int(length/step0)+1

        xstep = (b2[0]-b1[0])/K
        ystep = (b2[1]-b1[1])/K
        zstep = (b2[2]-b1[2])/K

        for k in range(1,K+1):
            x = b1[0]+k*xstep
            y = b1[1]+k*ystep
            z = b1[2]+k*zstep
            d = math.sqrt(x*x+y*y+z*z);
            if d > rad_thres:
                print "invalid VOID!!"
                break

            m += math.pow(x,p) * math.pow(y,q) * math.pow(z,r)
    return m

########################################################################
########################################################################
# Morphology calculations

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

Width=0.0
Height=0.0
Depth=0.0
Diameter=0.0
Length=0.0
Volume=0.0
Surface=0.0
Hausdorff=0.0
#
N_node=0
N_stem=0
N_bifs=0
N_branch=0
N_tips=0
Max_Order=0
#
Pd_ratio=0.0
Contraction=0.0
Max_Eux=0.0
Max_Path=0.0
BifA_local=0.0
BifA_remote=0.0
Soma_surface=0.0
Fragmentation=0.0
#
root_node=0

def computeFeature(nt):
    global Width, Height, Depth, Diameter, Length, Volume, Surface, Hausdorff
    global N_node, N_stem, N_bifs, N_branch, N_tips, Max_Order
    global Pd_ratio, Contraction, Max_Eux, Max_Path, BifA_local, BifA_remote, Soma_surface, Fragmentation
    global root_node
    # v3d returns a vector of 21 features. this is returned to the calling
    #   functin
    Width=0.0
    Height=0.0
    Depth=0.0
    Diameter=0.0
    Length=0.0
    Volume=0.0
    Surface=0.0
    Hausdorff=0.0
    #
    N_node=0
    N_stem=0
    N_bifs=0
    N_branch=0
    N_tips=0
    Max_Order=0
    #
    Pd_ratio=0
    Contraction=0
    Max_Eux=0
    Max_Path=0
    BifA_local=0
    BifA_remote=0
    Soma_surface=0
    Fragmentation=0
    root_node = nt.root

    N_node = nt.num_nodes
    N_stem = len(root_node[NODE_CHILDREN])
    Soma_surface = 4*math.pi*(root_node[NODE_R])*(root_node[NODE_R])

    computeLinear(nt)
    computeTree(nt)

    features = np.zeros(22)
    features[0] = N_node        # feature # 0: Number of Nodes          OK
    features[1] = Soma_surface  # feature # 1: Soma Surface             OK
    features[2] = N_stem        # feature # 2: Number of Stems          OK
    features[3] = N_bifs        # feature # 3: Number of Bifurcations   OK
    features[4] = N_branch      # feature # 4: Number of Branches       OK
    features[5] = N_tips        # feature # 5: Number of Tips           OK
    # Width, Height, Depth re-ordered relative to Xiaoxiao's data
    features[6] = Width         # feature # 6: Overall Width            ?
    features[7] = Height        # feature # 7: Overall Height           ?
    features[8] = Depth         # feature # 8: Overall Depth            ?
    features[9] = Diameter      # feature # 9: Average Diameter         OK
    features[10] = Length       # feature # 10: Total Length            OK
    features[11] = Surface      # feature # 11: Total Surface           OK
    features[12] = Volume       # feature # 12: Total Volume            OK
    features[13] = Max_Eux      # feature # 13: Max Euclidean Distance  OK
    features[14] = Max_Path     # feature # 14: Max Path Distance       OK
    features[15] = Max_Order    # feature # 15: Max Branch Order        OK
    features[16] = Contraction  # feature # 16: Average Contraction     OK
    features[17] = Fragmentation    #feature # 17: Average Fragmentation
    features[18] = Pd_ratio     # feature # 18: Average Parent-daughter Ratio
    features[19] = BifA_local   # feature # 19: Average Bifurcation Angle Local
    features[20] = BifA_remote  # feature # 20: Average Bifurcation Angle Remote
    if N_branch == 0:
         features[21] = float('nan')
    else:
         features[21] = 1.0 * N_node / N_branch

    feature_desc = []
    feature_desc.append("number_of_nodes")
    feature_desc.append("soma_surface")
    feature_desc.append("number_of_stems")
    feature_desc.append("number_of_bifurcations")
    feature_desc.append("number_of_branches")
    feature_desc.append("number_of_tips")
    feature_desc.append("overall_width")
    feature_desc.append("overall_height")
    feature_desc.append("overall_depth")
    feature_desc.append("average_diameter")
    feature_desc.append("total_length")
    feature_desc.append("total_surface")
    feature_desc.append("total_volume")
    feature_desc.append("max_euclidean_distance")
    feature_desc.append("max_path_distance")
    feature_desc.append("max_branch_order")
    feature_desc.append("average_contraction")
    feature_desc.append("average_fragmentation")
    feature_desc.append("average_parent_daughter_ratio")
    feature_desc.append("average_bifurcation_angle_local")
    feature_desc.append("average_bifurcation_angle_remote")
    feature_desc.append("nodes_over_branches")
    
    return features, feature_desc


# move forward and find indices (IDs) of furthest nodes along each
#   branch that are leaf nodes or that are right before the next
#   bifurcation
def getRemoteChild(nt, t):
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

# do a search along the list to compute overall N_bif, 
#   N_tip, width, height, depth, length, volume, surface, 
#   average diameter and max euclidean distance.
def computeLinear(nt):
    global Width, Height, Depth, Diameter, Length, Volume, Surface, Hausdorff
    global N_node, N_stem, N_bifs, N_branch, N_tips, Max_Order
    global Pd_ratio, Contraction, Max_Eux, Max_Path, BifA_local, BifA_remote, Soma_surface, Fragmentation
    global root_node
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
            N_tips += 1
        elif len(curr[NODE_CHILDREN]) > 1:
            N_bifs += 1
        parent = nt.parent_of(curr)
        if parent is None:
            continue
        l = dist(curr,parent);
        Diameter += 2*curr[NODE_R];
        Length += l;
        Surface += 2*math.pi*curr[NODE_R]*l
        Volume += math.pi*curr[NODE_R]*curr[NODE_R]*l
        lsoma = dist(curr, root_node)
        Max_Eux = max(Max_Eux,lsoma)
    Width = xmax-xmin
    Height = ymax-ymin
    Depth = zmax-zmin
    Diameter /= nt.num_nodes


# do a search along the tree to compute N_branch, max path distance, 
#   max branch order, average Pd_ratio, average Contraction, 
#   average Fragmentation, average bif angle local & remote
def computeTree(nt):
    global Width, Height, Depth, Diameter, Length, Volume, Surface, Hausdorff
    global N_node, N_stem, N_bifs, N_branch, N_tips, Max_Order
    global Pd_ratio, Contraction, Max_Eux, Max_Path, BifA_local, BifA_remote, Soma_surface, Fragmentation
    global root_node

    pathTotal = np.zeros(nt.num_nodes)
    depth = np.zeros(nt.num_nodes)

    stack = []
    stack.append(root_node[NODE_ID])
    pathlength = 0.0
    eudist = 0.0
    max_local_ang = 0.0
    max_remote_ang = 0.0
    
    N_ratio = 0
    N_Contraction = 0
   
    if len(root_node[NODE_CHILDREN]) > 1:
        local_ang = 0.0
        remote_ang = 0.0
        max_local_ang = 0.0
        max_remote_ang = 0.0
        ch_local1 = root_node[NODE_CHILDREN][0][NODE_ID]
        ch_local2 = root_node[NODE_CHILDREN][1][NODE_ID]
        local_ang = angle(root_node,nt.node(ch_local1),nt.node(ch_local2))

        remotes = getRemoteChild(nt, root_node)
        ch_remote1 = remotes[0]
        ch_remote2 = remotes[1]
        remote_ang = angle(root_node,nt.node(ch_remote1),nt.node(ch_remote2));
        if local_ang == local_ang:
            max_local_ang = max(max_local_ang,local_ang);
        if remote_ang == remote_ang:
            max_remote_ang = max(max_remote_ang,remote_ang);

        BifA_local += max_local_ang
        BifA_remote += max_remote_ang

    t = None
    tmp = None
    fragment = None
    while len(stack) > 0:
        t = stack.pop()
        for child in nt.node(t)[NODE_CHILDREN]:
            N_branch += 1
            if nt.node(t)[NODE_R] > 0:
                N_ratio += 1
                Pd_ratio += 1.0*child[NODE_R]/nt.node(t)[NODE_R]
            pathlength = dist(child, nt.node(t))

            fragment = 0.0
            child_id = child[NODE_ID]
            while len(nt.node(child_id)[NODE_CHILDREN]) == 1:
                ch = nt.node(child_id)[NODE_CHILDREN][0][NODE_ID]
                pathlength += dist(nt.node(ch), nt.node(child_id))
                fragment += 1
                child_id = ch
            eudist = dist(nt.node(child_id), nt.node(t))
            Fragmentation += fragment
            if pathlength > 0:
                Contraction += eudist/pathlength
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

                remotes = getRemoteChild(nt, child_id)
                ch_remote1 = remotes[0]
                ch_remote2 = remotes[1]
                remote_ang = angle(nt.node(child_id),nt.node(ch_remote1),nt.node(ch_remote2));
                if local_ang == local_ang:
                    max_local_ang = max(max_local_ang,local_ang)
                if remote_ang == remote_ang:
                    max_remote_ang = max(max_remote_ang,remote_ang)
                BifA_local += max_local_ang
                BifA_remote += max_remote_ang

            pathTotal[child_id] = pathTotal[t] + pathlength
            depth[child_id] = depth[t] + 1
    if N_ratio == 0:
        Pd_ratio = float('nan')
    else:
         Pd_ratio /= N_ratio

    if  N_branch == 0 :
       Fragmentation = float('nan')
    else:
      Fragmentation /= N_branch

    if N_Contraction == 0:
       Contraction = float('nan')
    else:
       Contraction /= N_Contraction
    
    if N_bifs==0:
        BifA_local = 0
        BifA_remote = 0
    else:
        BifA_local /= N_bifs
        BifA_remote /= N_bifs

    for i in range(nt.num_nodes):
        Max_Path = max(Max_Path, pathTotal[i])
        Max_Order = max(Max_Order, depth[i])

