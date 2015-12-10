#!/usr/bin/python
import morphology_analysis as morphology
from morphology_analysis_bb import compute_features as compute_features_bb
import copy
import sys
import psycopg2
import psycopg2.extras

def usage():
    print("This script calculates features from neuron morphology data")
    print("Input to the script is one or more specimen IDs")
    print("The script pulls the latest .swc file from LIMS, applies an")
    print("   affine transform to pia space and then calculates features")
    print("   for apical and basal dendrites")
    print("Output is formatted CSV");
    print("Format for input file must be a text file with one specimen id")
    print("   (or specimen name) per line")
    print("")
    if sys.argv[0].startswith("./"):
        name = sys.argv[0][2:]
    else:
        name = sys.argv[0]
    print("Usage: %s <-i <specimen_id> | -f <input_file> | -n <specimen_name>> <output file>" % name)
    print("with")
    print("   -i <specimen_id>   the id of a single specimen")
    print("   -f <input_file>    name of file containing list of specimen IDs")
    print("   -n <specimen_name> the name of a single specimen")
    sys.exit(1)

def parse_command_line():
    cmds = {}
    argc = len(sys.argv)
    i = 1
    while i < argc:
        tok = sys.argv[i]
        if tok[0] == '-':
            if i == argc-1:
                print("No token following flag %s" % tok)
                print("")
                usage()
            if tok[1] == 'i':
                i += 1
                if "specimen_id" not in cmds:
                    cmds["specimen_id"] = []
                cmds["specimen_id"].append(sys.argv[i])
            elif tok[1] == 'f':
                i += 1
                if "input_file" in cmds:
                    usage() # only one input file supported right now
                cmds["input_file"] = sys.argv[i]
            if tok[1] == 'n':
                i += 1
                if "specimen_name" not in cmds:
                    cmds["specimen_name"] = []
                cmds["specimen_name"].append(sys.argv[i])
        elif "output_file" not in cmds:
            cmds["output_file"] = tok
        else:
            print("Output file specified twice")
            print("")
            usage()
        i += 1
    if "output_file" not in cmds:
        print("No output file specified")
        print("")
        usage()
    if "specimen_id" not in cmds and "input_file" not in cmds and "specimen_name" not in cmds:
        print("No specimen ID/Name or input file specified")
        print("")
        usage()
    return cmds

# opens input file and reads each line into one of two arrays
# the id array is used if the entry is numeric (assuming that it's
#   a specimen id) and the name array is used otherwise (assuming
#   that it's a specimen name)
def read_input_file(fname):
    specimen_ids = []
    specimen_names = []
    try:
        f = open(fname, "r")
    except:
        print("Unable to open input file '%s'" % fname)
        sys.exit(1)
    try:
        content = f.readlines()
        for i in range(len(content)):
            line = content[i].rstrip()
            if len(line) == 0 or line[0] == '#':
                continue
            # try to guess if this is a specimen id or a specimen name
            try:
                specimen_id = int(line)
                specimen_ids.append(specimen_id)
            except:
                specimen_names.append(line)
    except:
        print("Error reading/parsing input file '%s'" % fname)
        raise
        sys.exit(1)
    f.close()
    return specimen_ids, specimen_names


########################################################################
# possible SQL queries to use
name_sql = ""
name_sql += "SELECT cell.id, cell.name, wkf.filename, wkf.storage_directory "
name_sql += "FROM specimens cell "
name_sql += "JOIN neuron_reconstructions nr ON cell.id = nr.specimen_id "
name_sql += "JOIN well_known_files wkf ON nr.id = wkf.attachable_id "
name_sql += "  AND wkf.attachable_type = 'NeuronReconstruction' "
name_sql += "WHERE cell.name='%s' AND nr.superseded = false; "

id_sql = ""
id_sql += "SELECT cell.id, cell.name, wkf.filename, wkf.storage_directory "
id_sql += "FROM specimens cell "
id_sql += "JOIN neuron_reconstructions nr ON cell.id = nr.specimen_id "
id_sql += "JOIN well_known_files wkf ON nr.id = wkf.attachable_id "
id_sql += "  AND wkf.attachable_type = 'NeuronReconstruction' "
id_sql += "WHERE cell.id=%s AND nr.superseded = false; "

aff_sql = ""
aff_sql += "SELECT "
aff_sql += "  a3d.tvr_00, a3d.tvr_01, a3d.tvr_02, "
aff_sql += "  a3d.tvr_03, a3d.tvr_04, a3d.tvr_05, "
aff_sql += "  a3d.tvr_06, a3d.tvr_07, a3d.tvr_08, "
aff_sql += "  a3d.tvr_09, a3d.tvr_10, a3d.tvr_11 "
aff_sql += "FROM specimens spc "
aff_sql += "JOIN neuron_reconstructions nr ON nr.specimen_id=spc.id "
aff_sql += "  AND nr.superseded = 'f' AND nr.manual = 't' "
aff_sql += "JOIN alignment3ds a3d ON a3d.id=spc.alignment3d_id "
aff_sql += "WHERE spc.id = %d;"

########################################################################
# database interface code
conn_string = "host='limsdb2' dbname='lims2' user='atlasreader' password='atlasro'"
conn = psycopg2.connect(conn_string)
cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

def fetch_specimen_record(sql):
    global cursor
    cursor.execute(sql)
    result = cursor.fetchall()
    spec_id = -1
    spec_name = ""
    name = ""
    path = ""
    record = {}
    if len(result) > 0:
        record["spec_id"] = result[0][0]
        record["spec_name"] = result[0][1]
        record["filename"] = result[0][2]
        record["path"] = result[0][3]
    return record

def fetch_affine_record(sql):
    global cursor
    cursor.execute(sql)
    result = cursor.fetchall()
    record = []
    if len(result) > 0:
        for i in range(12):
            record.append(1000.0 * result[0][i])
#        record["tvr_00"] = result[0][0]
#        record["tvr_01"] = result[0][1]
#        record["tvr_02"] = result[0][2]
#        record["tvr_03"] = result[0][3]
#        record["tvr_04"] = result[0][4]
#        record["tvr_05"] = result[0][5]
#        record["tvr_06"] = result[0][6]
#        record["tvr_07"] = result[0][7]
#        record["tvr_08"] = result[0][8]
#        record["tvr_09"] = result[0][9]
#        record["tvr_10"] = result[0][10]
#        record["tvr_11"] = result[0][11]
    print record
    print "-----------------------------------------"
    return record

########################################################################
# load input data

cmds = parse_command_line()
if "input_file" in cmds:
    id_list, name_list = read_input_file(cmds["input_file"])
else:
    id_list = []
    name_list = []

# merge command-line specified IDs with those in the input file
if "specimen_id" in cmds:
    for i in range(len(cmds["specimen_id"])):
        id_list.append(cmds["specimen_id"][i])

# merge command-line specified names with those in the input file
if "specimen_name" in cmds:
    for i in range(len(cmds["specimen_name"])):
        name_list.append(cmds["specimen_name"][i])

# for each id/name, query database to get file, path and name/id
records = {}
for i in range(len(id_list)):
    rec = fetch_specimen_record(id_sql % id_list[i])
    if len(rec) == 0:
        print("** Unable to read data for specimen ID '%s'" % id_list[i])
    else:
        records[rec["spec_name"]] = rec
for i in range(len(name_list)):
    rec = fetch_specimen_record(name_sql % name_list[i])
    if len(rec) == 0:
        print("** Unable to read data for '%s'" % name_list[i])
    else:
        # overwrite existing record if it's present
        records[rec["spec_name"]] = rec

########################################################################
# calculate features

# keep master dictionary of reported features
v3d_features = {}
bb_features = {}

# returns dictionary containing GMI and features
def calculate_v3d_features(morph, swc_type, label):
    global v3d_features
    cnt = 0
    soma_cnt = 0
    root_cnt = 0
    # strip out everything but the soma and the specified swc type
    for i in range(len(morph.obj_list)):
        obj = morph.obj_list[i]
        if obj.t == 1:
            soma_cnt += 1
        elif obj.t != swc_type:
            morph.obj_list[i] = None
        else:
            cnt += 1
        if obj.pn < 0:
            root_cnt += 1
    if cnt == 0:
        return None
    # v3d assumes there's only one root object. calculations can be
    #   erroneous if more exist
#    if soma_cnt != 1:
#        print("** Multiple somas detected. Skipping %s analysis to avoid errors" % label)
#        return None
    if root_cnt != 1:
        print("** Non-singular root detected. Skipping %s analysis" % label)
        return None
    # re-hash object tree
    morph.clean_up()
    # calculate features
    morph_data = {}
    try:
        gmi, gmi_desc = morphology.computeGMI(nrn)
        gmi_out = {}
        for j in range(len(gmi)):
            gmi_out[gmi_desc[j]] = gmi[j]
            if gmi_desc[j] not in v3d_features:
                v3d_features[gmi_desc[j]] = gmi_desc[j]
        morph_data["gmi"] = gmi_out
    except:
        print("Error calculating GMI for " + label)
        raise
    try:
        features, feature_desc = morphology.computeFeature(nrn)
        feat_out = {}
        for j in range(len(features)):
            feat_out[feature_desc[j]] =  features[j]
            if feature_desc[j] not in v3d_features:
                v3d_features[feature_desc[j]] = feature_desc[j]
        morph_data["features"] = feat_out
    except:
        print("Error calculating l-measure for " + label)
        raise
    return morph_data

# extract feature data
# global dictionary to store features. one entry per specimen_id
morph_data = {}
for k, record in records.iteritems():
    # get SWC file
    swc_file = record["path"] + record["filename"]
    print("Processing '%s'" % swc_file)
    try:
        nrn = morphology.SWC(swc_file)
        basal = morphology.SWC(swc_file)
        apical = morphology.SWC(swc_file)
    except Exception, e:
        print e
        print("Error -- unable to open specified file. Bailing out")
        print("---------------------------------------------------")
        print("Specimen id:   %d" % record["spec_id"])
        print("Specimen name: " + record["spec_name"])
        print("Specimen path: " + record["path"])
        print("Specimen file: " + record["filename"])
        raise
    # apply affine transform
    aff = fetch_affine_record(aff_sql % record["spec_id"])
    nrn.apply_affine(aff)
    basal.apply_affine(aff)
    #apical.apply_affine(aff)
    # save tmp swc file for BB library, using transformed coordinates
    tmp_swc_file = record["filename"] + ".tmp.swc"
    success = nrn.save_to(tmp_swc_file)
    #
    ####################################################################
    # v3d feature set
    #
    # strip axons from SWC
    for i in range(len(nrn.obj_list)):
        obj = nrn.obj_list[i]
        if obj.t == 2:
            nrn.obj_list[i] = None
    nrn.clean_up()
    basal_data = calculate_v3d_features(basal, 3, "basal dendrite")
    apical_data = calculate_v3d_features(apical, 3, "orig dendrite")
    #apical_data = calculate_v3d_features(apical, 4, "apical dendrite")
    data = {}
    if basal_data is not None:
        data["v3d_basal"] = basal_data
    if apical_data is not None:
        data["v3d_apical"] = apical_data
    ####################################################################
    # v3d feature set
    #
    # TODO check success value above to know what file to use
#    try:
#        bb_feat, bb_desc = compute_features_bb(tmp_swc_file)
#        feat_out = {}
#        for j in range(len(bb_feat)):
#            feat_out[bb_desc[j]] =  bb_feat[j]
#            if bb_desc[j] not in bb_features:
#                bb_features[bb_desc[j]] = bb_desc[j]
#        data["bb_features"] = feat_out
#    except:
#        print("Error calculating BB features")
#        raise
    morph_data[record["spec_name"]] = data

########################################################################
# sort data for better presentation

# make a sorted list of each feature set
v3d_feature_list = []
for k in v3d_features.keys():
    v3d_feature_list.append(k)
v3d_feature_list.sort()
bb_feature_list = []
for k in bb_features.keys():
    bb_feature_list.append(k)
bb_feature_list.sort()

# make a sorted list of specimen names
record_list = []
for k in records.keys():
    record_list.append(k)
record_list.sort()

########################################################################
# write output to csv file
try:
    f = open(cmds["output_file"], "w")
except IOError:
    print("Unable to open input file '%s'" % cmds["output_file"])
    sys.exit(1)
# write CSV header row
f.write("specimen_id,specimen_name,filename,")
for i in range(len(v3d_feature_list)):
    f.write("basal_" + v3d_feature_list[i] + ",")
for i in range(len(v3d_feature_list)):
    f.write("apical_" + v3d_feature_list[i] + ",")
#
#for i in range(len(bb_feature_list)):
#    f.write(bb_feature_list[i] + ",")
f.write("ignore\n")
import json
feat = {}
feat["feature_data"] = morph_data
feat["v3d_feature_list"] = v3d_feature_list
with open("out.json", "w") as jf:
    json.dump(feat, jf, indent=2)
    jf.close()

# write data
try:
    for i in range(len(record_list)):
        record = records[record_list[i]]
        spec_name = record["spec_name"]
        spec_id = record["spec_id"]
        filename = record["path"] + record["filename"]
        f.write("%s,%d,%s," % (spec_name, spec_id, filename))
        data = morph_data[spec_name]
        # v3d features
        # create an alias
        v3d = v3d_feature_list
        # basal dendrite
        if "v3d_basal" in data:
            for i in range(len(v3d)):
                if v3d[i] in data["v3d_basal"]["gmi"]:
                    val = str(data["v3d_basal"]["gmi"][v3d[i]])
                elif v3d[i] in data["v3d_basal"]["features"]:
                    val = str(data["v3d_basal"]["features"][v3d[i]])
                else:
                    val = "NaN"
                f.write(val + ",")
        else:
            for i in range(len(v3d)):
                f.write("NaN,")
        # apical dendrite
        if "v3d_apical" in data:
            for i in range(len(v3d)):
                if v3d[i] in data["v3d_apical"]["gmi"]:
                    val = str(data["v3d_apical"]["gmi"][v3d[i]])
                elif v3d[i] in data["v3d_apical"]["features"]:
                    val = str(data["v3d_apical"]["features"][v3d[i]])
                else:
                    val = "NaN"
                f.write(val + ",")
        else:
            for i in range(len(v3d)):
                f.write("NaN,")
        f.write("\n")
    f.close()
except IOError, ioe:
    print("File error encountered writing output file")
    print(ioe)
    sys.exit(1)

