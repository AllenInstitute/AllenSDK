import math
import sys
import numpy as np
#import allensdk.core.swc as allen_swc

# The code below is an almost literal c++->python port from the v3d 
#   blastneuron module (compute_gmi.cpp and compute_morph.cpp)
# The primary modification is that feature arrays are returned
#   from the functions (as opposed to being passed in) and 
#   a description (text) array is returned with each feature vector
# The SWC object mimics the interface of NeuronSWC in v3d. It has been
#   extended to assist in analysis tasks

class SWC_Obj(object):
    def __init__(self, obj):
        self.n = int(obj["id"])
        self.t = int(obj["type"])
        self.x = float(obj["x"])
        self.y = float(obj["y"])
        self.z = float(obj["z"])
        self.radius = float(obj["radius"])
        self.pn = int(obj["parent"])
        # create index-agnostic links between objects
        self.parent = None
        self.children = []

class SWC(object):
    def __init__(self, fname):
        # use allensdk if it's available
        self.obj_list = []
        self.obj_hash = {}
        try:
            import xallensdk.core.swc
            morphology = allen_swc.read_swc(fname)
            lst = morphology.compartment_list
            for i in range(len(lst)):
                obj = SWC_Obj(lst[i])
                self.obj_list.append(obj)
                self.obj_hash[obj.n] = len(self.obj_list) - 1
        except ImportError:
            f = open(fname, "r")
            line = f.readline()
            while len(line) > 0:
                if not line.startswith('#'):
                    toks = line.split(' ')
                    vals = {}
                    vals["id"] = int(toks[0])
                    vals["type"] = int(toks[1])
                    vals["x"] = float(toks[2])
                    vals["y"] = float(toks[3])
                    vals["z"] = float(toks[4])
                    vals["radius"] = float(toks[5])
                    pn = toks[6].strip('\r')
                    vals["parent"] = int(pn.strip('\n'))
                    #
                    obj = SWC_Obj(vals)
                    self.obj_list.append(obj)
                    self.obj_hash[obj.n] = len(self.obj_list) - 1
                    #self.obj_hash[obj.n] = obj
                line = f.readline()
            f.close()
        for i in range(len(self.obj_list)):
            obj = self.obj_list[i]
            if obj.pn >= 0:
                obj.parent = self.obj_list[self.obj_hash[obj.pn]]
                obj.parent.children.append(obj)
    
    # remove blank entries from obj_list and regenerate obj_hash
    # BB library requires SWC files that have no 'holes' in them
    def clean_up(self):
        # assign consecutive job IDs
        tmp_list = []
        n = 1
        for i in range(len(self.obj_list)):
            obj = self.obj_list[i]
            if obj is not None:
                obj.n = n
                n += 1
                tmp_list.append(obj)
        self.obj_list = tmp_list
        # re-link objects with parents
        for i in range(len(self.obj_list)):
            obj = self.obj_list[i]
            if obj.pn >= 0:
                obj.pn = obj.parent.n

    def apply_affine(self, aff):
        # calculate scale. use 2 different approaches
        #   1) assume isotropic spatial transform, use determinant^1/3
        #   2) calculate transform of unit vector on each original axis
        # (1)
        # calculate the determinant
        det0 = aff[0] * (aff[4]*aff[8] - aff[5]*aff[7])
        det1 = aff[1] * (aff[3]*aff[8] - aff[5]*aff[6])
        det2 = aff[2] * (aff[3]*aff[7] - aff[4]*aff[6])
        det = det0 + det1 + det2
        # determinant is change of volume that occurred during transform
        # assume equal scaling along all axes. take 3rd root to get
        #   scale factor
        det_scale = math.pow(abs(det), 1.0/3.0)
        # (2)
        scale_x = abs(aff[0] + aff[3] + aff[6])
        scale_y = abs(aff[1] + aff[4] + aff[7])
        scale_z = abs(aff[2] + aff[5] + aff[8])
        avg_scale = (scale_x + scale_y + scale_z) / 3.0;
        deviance = 0.0
        if scale_x > avg_scale:
            deviance = max(deviance, scale_x/avg_scale-1.0)
        else:
            deviance = max(deviance, 1.0-scale_x/avg_scale)
        if scale_y > avg_scale:
            deviance = max(deviance, scale_y/avg_scale-1.0)
        else:
            deviance = max(deviance, 1.0-scale_y/avg_scale)
        if scale_z > avg_scale:
            deviance = max(deviance, scale_z/avg_scale-1.0)
        else:
            deviance = max(deviance, 1.0-scale_z/avg_scale)
        # 
        for i in range(len(self.obj_list)):
            obj = self.obj_list[i]
            x = obj.x*aff[0] + obj.y*aff[1] + obj.z*aff[2] + aff[9]
            y = obj.x*aff[3] + obj.y*aff[4] + obj.z*aff[5] + aff[10]
            z = obj.x*aff[6] + obj.y*aff[7] + obj.z*aff[8] + aff[11]
            obj.x = x
            obj.y = y
            obj.z = z
            # use method (1) for scaling for now as it's most simple
            obj.radius *= det_scale

    # returns True on success, False on failure
    def save_to(self, file_name):
        try:
            f = open(file_name, "w")
            f.write("#n,type,x,y,z,radius,parent\n")
            for i in range(len(self.obj_list)):
                obj = self.obj_list[i]
                f.write("%d %d %f " % (obj.n, obj.t, obj.x))
                f.write("%f %f %f %d\n" % (obj.y, obj.z, obj.radius, obj.pn))
            f.close()
        except:
            print("Error creating swc file '%s'" % file_name)
            return False
        return True

########################################################################
########################################################################
# GMI calculations

VOID = 1000000000

def computeGMI(nt):
    # v3d version calculates 14 moments
    gmi = np.zeros(14)
    lst = nt.obj_list
    LUT = {}
    for i in range(len(lst)):
        LUT[lst[i].n] = i

    center_pos = np.zeros(3)
    
    siz = len(lst)
    b = []
    for i in range(siz):
        b.append([])

    avgR = 0.0;

    for i in range(siz):
        b[i] = np.zeros(4)
        b[i][0] = lst[i].x
        b[i][1] = lst[i].y
        b[i][2] = lst[i].z
        avgR += lst[i].radius
        if lst[i].pn < 0:
            b[i][3] = -1;
        else:
            b[i][3] = LUT[lst[i].pn];

    avgR /= siz
    gmi[13] = avgR

    m000 = compute_moments(b, siz, 0, 0, 0, VOID);
    center_pos[0] = compute_moments(b, siz, 1, 0, 0, VOID);
    center_pos[1] = compute_moments(b, siz, 0, 1, 0, VOID);
    center_pos[2] = compute_moments(b, siz, 0, 0, 1, VOID);

    for j in range(3):
        center_pos[j] /= m000
    compute_neuron_GMI(b, siz, center_pos, VOID, gmi);

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
    return math.sqrt(((a).x-(b).x)*((a).x-(b).x)+((a).y-(b).y)*((a).y-(b).y)+((a).z-(b).z)*((a).z-(b).z))

def getParent(n,nt): 
    if nt.obj_list[n].pn < 0:
        return VOID
    else:
        return nt.obj_hash[nt.obj_list[n].pn]

def angle(a,b,c):
    dist_ab = dist(a, b)
    dist_ac = dist(a, c)
    if dist_ab == 0 or dist_ac == 0:        
        print ("Warning. Parent and child SWC nodes have same coordinates. No bifurcation angle to compute.")
        print ("Parent at  %f,%f,%f" % (a.x, a.y, a.z))
        print ("Child 0 at %f,%f,%f" % (b.x, b.y, b.z))
        print ("Child 1 at %f,%f,%f" % (c.x, c.y, c.z))
        return float('nan')
    return(math.acos((((b).x-(a).x)*((c).x-(a).x)+((b).y-(a).y)*((c).y-(a).y)+((b).z-(a).z)*((c).z-(a).z))/(dist_ab*dist_ac))*180.0/math.pi)

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
rootidx=0

childs = []

def computeFeature(nt):
    global Width, Height, Depth, Diameter, Length, Volume, Surface, Hausdorff
    global N_node, N_stem, N_bifs, N_branch, N_tips, Max_Order
    global Pd_ratio, Contraction, Max_Eux, Max_Path, BifA_local, BifA_remote, Soma_surface, Fragmentation
    global childs, rootidx
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
    #
    rootidx=0

    childs = []
    for i in range(len(nt.obj_list)):
        par = nt.obj_list[i].pn
        if par >= 0:
            pnum = nt.obj_hash[par]
            while len(childs) <= pnum:
                childs.append([])
            childs[pnum].append(i)
    while len(childs) < len(nt.obj_list):
        childs.append([])

    #find the root
    rootidx = VOID;
    lst = nt.obj_list
    for i in range(len(lst)):
        if lst[i].pn == -1:
            if rootidx != VOID:
                print "WARNING - multiple roots are specified. Using the latter"
            rootidx = i
    if rootidx == VOID:
        print "the input neuron tree does not have a root, please check your data"
        return

    N_node = len(lst)
    N_stem = len(childs[rootidx])
    Soma_surface = 4*math.pi*(lst[rootidx].radius)*(lst[rootidx].radius)

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


def getRemoteChild(t):
    global childs
    rchildlist = []
    for i in range(len(childs[t])):
        tmp = childs[t][i]
        while len(childs[tmp]) == 1:
            tmp = childs[tmp][0]
        rchildlist.append(tmp)
    return rchildlist

# do a search along the list to compute overall N_bif, 
#   N_tip, width, height, depth, length, volume, surface, 
#   average diameter and max euclidean distance.
def computeLinear(nt):
    global Width, Height, Depth, Diameter, Length, Volume, Surface, Hausdorff
    global N_node, N_stem, N_bifs, N_branch, N_tips, Max_Order
    global Pd_ratio, Contraction, Max_Eux, Max_Path, BifA_local, BifA_remote, Soma_surface, Fragmentation
    global childs, rootidx
    xmin = VOID
    ymin = VOID
    zmin = VOID
    xmax = 0.0
    ymax = 0.0
    zmax = 0.0
    lst = nt.obj_list
    soma = lst[rootidx]

    for i in range(len(lst)):
        curr = lst[i];
        xmin = min(xmin,curr.x)
        ymin = min(ymin,curr.y)
        zmin = min(zmin,curr.z)
        xmax = max(xmax,curr.x)
        ymax = max(ymax,curr.y)
        zmax = max(zmax,curr.z)
        if len(childs[i]) == 0:
            N_tips += 1
        elif len(childs[i]) > 1:
            N_bifs += 1
        parent = getParent(i,nt);
        if parent == VOID:
            continue
        try:
            l = dist(curr,lst[parent]);
        except IndexError:
            print parent
            print curr
            print len(lst)
            raise
        Diameter += 2*curr.radius;
        Length += l;
        Surface += 2*math.pi*curr.radius*l
        Volume += math.pi*curr.radius*curr.radius*l
        lsoma = dist(curr,soma)
        Max_Eux = max(Max_Eux,lsoma)
    Width = xmax-xmin
    Height = ymax-ymin
    Depth = zmax-zmin
    Diameter /= len(lst)


# do a search along the tree to compute N_branch, max path distance, 
#   max branch order, average Pd_ratio, average Contraction, 
#   average Fragmentation, average bif angle local & remote
def computeTree(nt):
    global Width, Height, Depth, Diameter, Length, Volume, Surface, Hausdorff
    global N_node, N_stem, N_bifs, N_branch, N_tips, Max_Order
    global Pd_ratio, Contraction, Max_Eux, Max_Path, BifA_local, BifA_remote, Soma_surface, Fragmentation
    global childs, rootidx
    lst = nt.obj_list

    pathTotal = np.zeros(len(lst))
    depth = np.zeros(len(lst))

    stack = []
    stack.append(rootidx)
    pathlength = 0.0
    eudist = 0.0
    max_local_ang = 0.0
    max_remote_ang = 0.0
    
    N_ratio = 0
    N_Contraction = 0
   
    if len(childs[rootidx]) > 1:
        local_ang = 0.0
        remote_ang = 0.0
        max_local_ang = 0.0
        max_remote_ang = 0.0
        ch_local1 = childs[rootidx][0]
        ch_local2 = childs[rootidx][1]
        local_ang = angle(lst[rootidx],lst[ch_local1],lst[ch_local2])

        ch_remote1 = getRemoteChild(rootidx)[0];
        ch_remote2 = getRemoteChild(rootidx)[1];
        remote_ang = angle(lst[rootidx],lst[ch_remote1],lst[ch_remote2]);
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
        child = childs[t]
        for i in range(len(child)):
            N_branch += 1
            tmp = child[i]
            if lst[t].radius > 0:
                N_ratio += 1
                Pd_ratio += lst[tmp].radius/lst[t].radius
            pathlength = dist(lst[tmp],lst[t])

            fragment = 0.0
            while len(childs[tmp]) == 1:
                ch = childs[tmp][0]
                pathlength += dist(lst[ch],lst[tmp])
                fragment += 1
                tmp = ch
            eudist = dist(lst[tmp],lst[t])
            Fragmentation += fragment
            if pathlength > 0:
                Contraction += eudist/pathlength
                N_Contraction += 1

            #we are reaching a tip point or another branch point, 
            #   computation for this branch is over
            chsz = len(childs[tmp])
            if chsz > 1:  #another branch
                stack.append(tmp)

                #compute local bif angle and remote bif angle
                local_ang = 0.0
                remote_ang = 0.0
                max_local_ang = 0
                max_remote_ang = 0
                ch_local1 = childs[tmp][0]
                ch_local2 = childs[tmp][1]
                local_ang = angle(lst[tmp],lst[ch_local1],lst[ch_local2])

                ch_remote1 = getRemoteChild(tmp)[0];
                ch_remote2 = getRemoteChild(tmp)[1];
                remote_ang = angle(lst[tmp],lst[ch_remote1],lst[ch_remote2]);
                if local_ang == local_ang:
                    max_local_ang = max(max_local_ang,local_ang)
                if remote_ang == remote_ang:
                    max_remote_ang = max(max_remote_ang,remote_ang)

                BifA_local += max_local_ang
                BifA_remote += max_remote_ang

            pathTotal[tmp] = pathTotal[t] + pathlength
            depth[tmp] = depth[t] + 1
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

    for i in range(len(lst)):
        Max_Path = max(Max_Path, pathTotal[i])
        Max_Order = max(Max_Order, depth[i])

