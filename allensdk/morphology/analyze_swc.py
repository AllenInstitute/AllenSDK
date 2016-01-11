#!/usr/bin/python
import morphology_analysis as morphology
#from morphology_analysis_bb import compute_features as compute_features_bb
from bb3 import compute_features as compute_features_bb
from bb3 import compute_keith_features
import traceback
import sys
import psycopg2
import psycopg2.extras

CALCULATE_AXONS = False


def usage():
    print("This script calculates features from neuron morphology data")
    print("Input to the script is one or more specimen IDs or names")
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
    print("Usage: %s <-a | -i <specimen_id> | -f <input_file> | -n <specimen_name>> <output file>" % name)
    print("with")
    print("  -a                select all morphologies from LIMS")
    print("  -i <specimen_id>  the id of a single specimen")
    print("  -f <input_file>   name of file containing list of specimen IDs or SWC files")
    print("  -n <specimen_name> the name of a single specimen")
    print("  -x                do not perform affine transform")
    sys.exit(1)

def parse_command_line():
    cmds = {}
    argc = len(sys.argv)
    i = 1
    cmds["perform_affine"] = True
    while i < argc:
        tok = sys.argv[i]
        if tok[0] == '-':
            if i == argc-1:
                print("No token following flag %s" % tok)
                print("")
                usage()
            if tok[1] == 'a':
                cmds["select_all"] = True
            elif tok[1] == 'i':
                i += 1
                if "specimen_id" not in cmds:
                    cmds["specimen_id"] = []
                cmds["specimen_id"].append(sys.argv[i])
            elif tok[1] == 'f':
                i += 1
                if "input_file" in cmds:
                    usage() # only one input file supported right now
                cmds["input_file"] = sys.argv[i]
            elif tok[1] == 'n':
                i += 1
                if "specimen_name" not in cmds:
                    cmds["specimen_name"] = []
                cmds["specimen_name"].append(sys.argv[i])
            elif tok[1] == 'x':
                cmds["perform_affine"] = False
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
    if "select_all" in cmds:
        if "specimen_id" in cmds or "input_file" in cmds or "specimen_name" in cmds:
            print("If specifying -a, cannot specify, -i, -f or -n")
            print("")
            usage()
    elif "specimen_id" not in cmds and "input_file" not in cmds and "specimen_name" not in cmds:
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
    file_names = []
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
                if line.endswith(".swc"):
                    file_names.append(line)
                else:
                    specimen_names.append(line)
    except:
        print("Error reading/parsing input file '%s'" % fname)
        raise
        sys.exit(1)
    f.close()
    return specimen_ids, specimen_names, file_names


########################################################################
# possible SQL queries to use

all_sql = ""
all_sql += "SELECT sp.id "
all_sql += "FROM specimens sp "
all_sql += "JOIN ephys_roi_results err on err.id = sp.ephys_roi_result_id "
all_sql += "JOIN neuron_reconstructions nr on nr.specimen_id = sp.id "
all_sql += "WHERE err.workflow_state = 'manual_passed' "
all_sql += "AND nr.superseded is false AND nr.manual is true "
all_sql += "ORDER by sp.name; "

base_sql = ""
base_sql += "with dendrite_type as  \n"
base_sql += "( \n"
base_sql += "  select sts.specimen_id, st.name  \n"
base_sql += "  from specimen_tags_specimens sts \n"
base_sql += "  join specimen_tags st on sts.specimen_tag_id = st.id \n"
base_sql += "  where st.name like 'dendrite type%s' \n"
base_sql += ") \n"

name_sql = base_sql
name_sql += "SELECT spec.id, spec.name, dt.name, str.name, wkf.filename, wkf.storage_directory \n"
name_sql += "FROM specimens spec \n"
name_sql += "LEFT JOIN structures str on spec.structure_id = str.id \n"
name_sql += "LEFT JOIN dendrite_type dt on dt.specimen_id = spec.id \n"
name_sql += "JOIN neuron_reconstructions nr ON nr.specimen_id=spec.id \n"
name_sql += "  AND nr.superseded = 'f' AND nr.manual = 't' \n"
name_sql += "JOIN well_known_files wkf ON wkf.attachable_id=nr.id \n"
name_sql += "  AND wkf.attachable_type = 'NeuronReconstruction' \n"
name_sql += "JOIN cell_soma_locations csl ON csl.specimen_id=spec.id \n"
name_sql += "JOIN well_known_file_types wkft \n"
name_sql += "  ON wkft.id=wkf.well_known_file_type_id \n"
name_sql += "WHERE spec.name='%s' AND wkft.name = '3DNeuronReconstruction'; \n"

id_sql = base_sql
id_sql += "SELECT spec.id, spec.name, dt.name, str.name, wkf.filename, wkf.storage_directory \n"
id_sql += "FROM specimens spec \n"
id_sql += "LEFT JOIN structures str on spec.structure_id = str.id \n"
id_sql += "LEFT JOIN dendrite_type dt on dt.specimen_id = spec.id \n"
id_sql += "JOIN neuron_reconstructions nr ON nr.specimen_id=spec.id \n"
id_sql += "  AND nr.superseded = 'f' AND nr.manual = 't' \n"
id_sql += "JOIN well_known_files wkf ON wkf.attachable_id=nr.id \n"
id_sql += "  AND wkf.attachable_type = 'NeuronReconstruction' \n"
id_sql += "JOIN cell_soma_locations csl ON csl.specimen_id=spec.id \n"
id_sql += "JOIN well_known_file_types wkft \n"
id_sql += "  ON wkft.id=wkf.well_known_file_type_id \n"
id_sql += "WHERE spec.id=%s AND wkft.name = '3DNeuronReconstruction'; \n"

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
try:
  conn_string = "host='limsdb2' dbname='lims2' user='atlasreader' password='atlasro'"
  conn = psycopg2.connect(conn_string)
  cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
except:
  print "no db connections!"

def fetch_all_swcs():
    global cursor, all_sql
    cursor.execute(all_sql)
    result = cursor.fetchall()
    id_list = []
    for i in range(len(result)):
        id_list.append(result[i][0])
    return id_list, []

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
        if result[0][2] is not None:
            record["dend_type"] = result[0][2].replace(","," ")
        else:
            record["dend_type"] = ""
        if result[0][3] is not None:
            record["location"] = result[0][3].replace(","," ")
        else:
            record["location"] = ""
        record["filename"] = result[0][4]
        record["path"] = result[0][5]
    return record

def fetch_affine_record(sql):
    global cursor
    cursor.execute(sql)
    result = cursor.fetchall()
    record = []
    if len(result) > 0:
        for i in range(12):
            record.append(1000.0 * result[0][i])
    #print record
    return record

########################################################################
# load input data

cmds = parse_command_line()
if "input_file" in cmds:
    id_list, name_list, file_list = read_input_file(cmds["input_file"])
elif "select_all" in cmds:
    id_list, name_list, file_list = fetch_all_swcs()
else:
    id_list = []
    name_list = []
    file_list = []

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
    rec = fetch_specimen_record(id_sql % ('%', id_list[i]))
    if len(rec) == 0:
        print("** Unable to read data for specimen ID '%s'" % id_list[i])
    else:
        records[rec["spec_name"]] = rec
for i in range(len(name_list)):
    rec = fetch_specimen_record(name_sql % ('%', name_list[i]))
    if len(rec) == 0:
        print("** Unable to read data for '%s'" % name_list[i])
    else:
        # overwrite existing record if it's present
        records[rec["spec_name"]] = rec
if len(file_list) > 0:
    if cmds["perform_affine"]:
        print("When SWC files are specified, affine transform must be disabled (flag -x)")
        sys.exit(1)
    for i in range(len(file_list)):
        rec = {}
        rec["spec_id"] = i
        rec["spec_name"] = file_list[i]
        rec["dend_type"] = ""
        rec["location"] = ""
        rec["filename"] = file_list[i]
        rec["path"] = ""
        records[file_list[i]] = rec

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
    # keep track of number of soma roots and roots in specified type
    for i in range(len(morph.obj_list)):
        obj = morph.obj_list[i]
        if obj.t == 1:
            soma_cnt += 1
            if obj.pn < 0:
                root_cnt += 1
        elif obj.t != swc_type:
            morph.obj_list[i] = None
        else:
            if obj.pn < 0:
                root_cnt += 1
            cnt += 1
    if cnt == 0:
        return None
    # v3d assumes there's only one root object. calculations can be
    #   erroneous if more exist
    if soma_cnt != 1:
        print("** Multiple somas detected. Skipping %s analysis to avoid errors" % label)
        print("***DEBUG -- Not skipping analysis***")
        #return None
    if root_cnt != 1:
        print("** Non-singular root detected. Skipping %s analysis" % label)
        return None
    # re-hash object tree
    morph.clean_up()
    # calculate features
    results = {}
    try:
        gmi, gmi_desc = morphology.computeGMI(morph)
        if gmi is None:
            return None
        gmi_out = {}
        for j in range(len(gmi)):
            gmi_out[gmi_desc[j]] = gmi[j]
            if gmi_desc[j] not in v3d_features:
                v3d_features[gmi_desc[j]] = gmi_desc[j]
        results["gmi"] = gmi_out
    except:
        print("Error calculating GMI for " + label)
        raise
    try:
        features, feature_desc = morphology.computeFeature(morph)
        if features is None:
            return None
        feat_out = {}
        for j in range(len(features)):
            feat_out[feature_desc[j]] =  features[j]
            if feature_desc[j] not in v3d_features:
                v3d_features[feature_desc[j]] = feature_desc[j]
        results["features"] = feat_out
    except:
        print("Error calculating l-measure for " + label)
        raise
    return results

# extract feature data
# global dictionary to store features. one entry per specimen_id
morph_data = {}

for k, record in records.iteritems():
    # get SWC file
    swc_file = record["path"] + record["filename"]
    print("Processing '%s'" % swc_file)
    try:
        nrn = morphology.SWC(swc_file)
        axon = morphology.SWC(swc_file)
        basal = morphology.SWC(swc_file)
        apical = morphology.SWC(swc_file)
    except Exception, e:
        #print e
        print("")
        print("**** Error: problem encountered open specified file ****")
        print("Specimen id:   %d" % record["spec_id"])
        print("Specimen name: " + record["spec_name"])
        print("Specimen path: " + record["path"])
        print("Specimen file: " + record["filename"])
        print("")
        print(traceback.print_exc())
        print("-----------------------------------------------------------")
        continue
    try:
        # apply affine transform
        if cmds["perform_affine"]:
            aff = fetch_affine_record(aff_sql % record["spec_id"])
            nrn.apply_affine(aff)
            # save a copy of affine-corrected file
            tmp_swc_file = record["filename"][:-4] + "_pia.swc"
            nrn.save_to(tmp_swc_file)
            # apply affine to basal and apical copies too
            if CALCULATE_AXONS:
                axon.apply_affine(aff)
            basal.apply_affine(aff)
            apical.apply_affine(aff)
        #
        ####################################################################
        # strip axons from SWC
        ####################################################################
        # v3d feature set
        #
        data = {}
        if CALCULATE_AXONS:
          print "calculate axon features"
          axon_data = calculate_v3d_features(axon, 2, "axon")
          if axon_data is not None:
              data["v3d_axon"] = axon_data

        print "calculate basal features"
        basal_data = calculate_v3d_features(basal, 3, "basal dendrite")
        if basal_data is not None:
            data["v3d_basal"] = basal_data

        print "calculate apical features"
        apical_data = calculate_v3d_features(apical, 4, "apical dendrite")
        if apical_data is not None:
            data["v3d_apical"] = apical_data
        ####################################################################
        # BB feature set
        #
        # write cleaned-up file for BB to use
        for i in range(len(nrn.obj_list)):
            obj = nrn.obj_list[i]
            if obj.t == 2:
                nrn.obj_list[i] = None
        nrn.clean_up()
        tmp_swc_file_bb = record["filename"][:-4] + "_pia_bb.swc"
        success = nrn.save_to(tmp_swc_file_bb)
        # calculate features
        try:
            bb_data, keith_data = compute_features_bb(tmp_swc_file_bb)
            compute_keith_features(nrn, keith_data, bb_data)
            data["bb_features"] = bb_data
            for k in bb_data:
                if k not in bb_features:
                    bb_features[k] = k
        except:
            print("Error calculating BB features")
            raise
    except Exception, e:
        print("")
        print("**** Error: analyzing file ****")
        print("Specimen id:   %d" % record["spec_id"])
        print("Specimen name: " + record["spec_name"])
        print("Specimen path: " + record["path"])
        print("Specimen file: " + record["filename"])
        print("")
        traceback.print_exc()
        print("-----------------------------------------------------------")
        continue
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
f.write("specimen_name,specimen_id,dendrite_type,region_info,filename,")
#for i in range(len(v3d_feature_list)):
#    f.write("axon_" + v3d_feature_list[i] + ",")
for i in range(len(v3d_feature_list)):
    f.write("basal_" + v3d_feature_list[i] + ",")
for i in range(len(v3d_feature_list)):
    f.write("apical_" + v3d_feature_list[i] + ",")
for i in range(len(bb_feature_list)):
    f.write(bb_feature_list[i] + ",")
f.write("ignore\n")

#import json
#feat = {}
#feat["feature_data"] = morph_data
#feat["v3d_feature_list"] = v3d_feature_list
#feat["v3d_feature_list"] = v3d_feature_list
#with open("out.json", "w") as jf:
#    json.dump(feat, jf, indent=2)
#    jf.close()

# write data
try:
    for i in range(len(record_list)):
        record = records[record_list[i]]
        spec_name = record["spec_name"]
        if spec_name not in morph_data:
            continue    # perhaps skipped due earlier error
        data = morph_data[spec_name]
        # error processing neuron -- omit from output
        if "v3d_basal" not in data and "v3d_apical" not in data:
            continue
        spec_id = record["spec_id"]
        dend_type = record["dend_type"]
        location = record["location"]
        filename = record["path"] + record["filename"]
        f.write("%s,%d,%s,%s,%s," % (spec_name, spec_id, dend_type, location, filename))
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
        # axon
        if CALCULATE_AXONS:
          if "v3d_axon" in data:
              for i in range(len(v3d)):
                  if v3d[i] in data["v3d_axon"]["gmi"]:
                      val = str(data["v3d_axon"]["gmi"][v3d[i]])
                  elif v3d[i] in data["v3d_axon"]["features"]:
                      val = str(data["v3d_axon"]["features"][v3d[i]])
                  else:
                      val = "NaN"
                  f.write(val + ",")
          else:
              for i in range(len(v3d)):
                  f.write("NaN,")
        # BB features
        if "bb_features" in data:
            bb = bb_feature_list
            for i in range(len(bb)):
                if bb[i] in data["bb_features"]:
                    val = str(data["bb_features"][bb[i]])
                else:
                    val = "NaN"
                f.write(val + ",")
        else:
            for i in range(len(bb)):
                f.write("NaN,")
        f.write("\n")
    f.close()
except IOError, ioe:
    print("File error encountered writing output file")
    print(ioe)
    sys.exit(1)

