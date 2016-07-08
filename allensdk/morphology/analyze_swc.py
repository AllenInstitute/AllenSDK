#!/usr/bin/python
import traceback
import sys
import psycopg2
import psycopg2.extras
import swc
from feature_extractor import *

import prep_upright

CALCULATE_AXONS = True


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
    print("  -c                fix type-inconstency error")
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
    cmds["fix_type_error"] = False
    while i < argc:
        tok = sys.argv[i]
        if tok[0] == '-':
            if tok[1] == 'a':
                cmds["select_all"] = True
            elif tok[1] == 'c':
                cmds["fix_type_error"] = True
            elif tok[1] == 'i':
                if i == argc-1:
                    print("No token following flag %s" % tok)
                    print("")
                    usage()
                i += 1
                if "specimen_id" not in cmds:
                    cmds["specimen_id"] = []
                cmds["specimen_id"].append(sys.argv[i])
            elif tok[1] == 'f':
                if i == argc-1:
                    print("No token following flag %s" % tok)
                    print("")
                    usage()
                i += 1
                if "input_file" in cmds:
                    usage() # only one input file supported right now
                cmds["input_file"] = sys.argv[i]
            elif tok[1] == 'n':
                if i == argc-1:
                    print("No token following flag %s" % tok)
                    print("")
                    usage()
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
    return id_list, [], []

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

def fetch_affine_record(specimen_id):
    global cursor
    return prep_upright.calculate_transform(cursor, specimen_id)

def fetch_affine_record_old(sql):
    global cursor
    cursor.execute(sql)
    result = cursor.fetchall()
    record = []
    if len(result) > 0:
        for i in range(12):
            record.append(1000.0 * result[0][i])
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

# returns dictionary containing GMI and features
def calculate_v3d_features(morph, swc_type, label):
    global v3d_features
    # strip out everything but the soma and the specified swc type
    morph.strip_all_other_types(swc_type)
    if len(morph.node_list_by_type(swc_type)) < 5:
        return
    print("calculate %s features" % label)
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
    # get SWC
    swc_file = record["path"] + record["filename"]
    print("Processing '%s'" % swc_file)
    try:
        nrn = swc.read_swc(swc_file)
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
    # process features
    try:
        # apply affine transform, if appropriate
        if cmds["perform_affine"]:
            aff = fetch_affine_record(record["spec_id"])
            #aff = fetch_affine_record_old(aff_sql % record["spec_id"])
            nrn.apply_affine(aff)
            #nrn.apply_affine_only_rotation(aff)
            ## save a copy of affine-corrected file
            tmp_swc_file = record["filename"][:-4] + "_pia.swc"
            nrn.write(tmp_swc_file)
        features = MorphologyFeatures(nrn)
        data = {}
        data["axon"] = features.axon
        data["cloud"] = features.axon_cloud
        data["dendrite"] = features.dendrite
        data["basal_dendrite"] = features.basal_dendrite
        data["apical_dendrite"] = features.apical_dendrite
        data["all_neurites"] = features.all_neurites
        morph_data[record["spec_name"]] = data
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

TYPES = ["axon", "cloud", "dendrite", "basal_dendrite", "apical_dendrite", "all_neurites"]

# make a sorted list of specimen names
record_list = []
for k in records.keys():
    record_list.append(k)
record_list.sort()


########################################################################
# write output to csv file(s)

def write_features(record_list, groups=None):
    # open the output file
    # sometimes only a subset of data is to be output (eg, axon only)
    # construct the filename to indicate which subset of data is contained
    #   in the file. where all data is desired, use the specified filename
    #   in its unaltered state
    try:
        suffix = ""
        # use the group list to determine which categories to output
        if groups == None:
            groups = TYPES
        else:
            # construct the suffix by concatenating all group names
            suffix = "."
            for i in range(len(groups)-1):
                suffix += groups[i] + "-"
            suffix += groups[-1]
        fname = cmds["output_file"]
        idx = fname.rfind('.')
        if idx > 0:
            outfile = fname[0:idx] + suffix + fname[idx:]
        else:
            outfile = fname
        print outfile
        f = open(outfile, "w")
    except IOError:
        print("Unable to open input file '%s'" % cmds["output_file"])
        sys.exit(1)
    # write CSV header row
    f.write("specimen_name,specimen_id,dendrite_type,region_info,filename,")
    # use one record to get data column names (any will do)
    for rec in record_list:
        record = records[rec]
        spec_name = record["spec_name"]
        for grp in groups:
            prefix = grp + "_"
            # make sorted list of features names
            names = []
            for k in morph_data[spec_name][grp].keys():
                names.append(k)
            names.sort()
            for j in range(len(names)):
                f.write(prefix + names[j] + ",")
        f.write("ignore\n")
        break

# write data
    try:
        for rec in record_list:
            record = records[rec]
            spec_name = record["spec_name"]
            if spec_name not in morph_data:
                continue    # perhaps skipped due earlier error
            spec_id = record["spec_id"]
            dend_type = record["dend_type"]
            location = record["location"]
            filename = record["path"] + record["filename"]
            f.write("%s,%d,%s,%s,%s," % (spec_name, spec_id, dend_type, location, filename))

            for grp in groups:
                prefix = grp + "_"
                data = morph_data[spec_name][grp]
                names = []
                for k in morph_data[spec_name][grp].keys():
                    names.append(k)
                names.sort()
                for name in names:
                    f.write(str(data[name]) + ",")
            f.write("\n")
        f.close()
    except IOError, ioe:
        print("File error encountered writing output file")
        print(ioe)
        sys.exit(1)

#write_features(record_list, ["axon"])
write_features(record_list, ["basal_dendrite"])
#write_features(record_list, ["basal_dendrite, apical_dendrite, dendrite"])
write_features(record_list)
