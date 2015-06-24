
# metadata storage constants
# these are so users don't have to remember exact path to
#   meatadata fields, and to avoid/detect typos

DATA_COLLECTION = "data_collection"
EXPERIMENT_DESCRIPTION = "experiment_description"
EXPERIMENTER = "experimenter"
INSTITUTION = "institution"
LAB = "lab"
NOTES = "notes"
PROTOCOL = "protocol"
PHARMACOLOGY = "pharmacology"
RELATED_PUBLICATIONS = "related_publications"
SESSION_ID = "session_id"
SLICES = "slices"
STIMULUS = "stimulus"
SURGERY = "surgery"
VIRUS = "virus"

SUBJECT = "subject/description"
SUBJECT_ID = "subject/subject_id"
SPECIES = "subject/species"
GENOTYPE = "subject/genotype"
SEX = "subject/sex"
AGE = "subject/age"
WEIGHT = "subject/weight"


EXTRA_ELECTRODE_MAP = "extracellular_ephys/electrode_map"
EXTRA_ELECTRODE_GROUP = "extracellular_ephys/electrode_group"
EXTRA_IMPEDANCE = "extracellular_ephys/impedance"
EXTRA_FILTERING = "extracellular_ephys/filtering"

def EXTRA_CUSTOM(name):
    return "extracellular_ephys/" + name
def EXTRA_SHANK_DESCRIPTION(shank):
    return "extracellular_ephys/" + shank + "/description"
def EXTRA_SHANK_LOCATION(shank):
    return "extracellular_ephys/" + shank + "/description"
def EXTRA_SHANK_DEVICE(shank):
    return "extracellular_ephys/" + shank + "/device"
def EXTRA_SHANK_CUSTOM(shank, name):
    return "extracellular_ephys/" + shank + "/device/" + name

def INTRA_ELECTRODE_DESCRIPTION(name):
    return "intracellular_ephys/" + name + "/description"
def INTRA_ELECTRODE_FILTERING(name):
    return "intracellular_ephys/" + name + "/filtering"
def INTRA_ELECTRODE_DEVICE(name):
    return "intracellular_ephys/" + name + "/device"
def INTRA_ELECTRODE_LOCATION(name):
    return "intracellular_ephys/" + name + "/location"
def INTRA_ELECTRODE_RESISTANCE(name):
    return "intracellular_ephys/" + name + "/resistance"
def INTRA_ELECTRODE_SLICE(name):
    return "intracellular_ephys/" + name + "/slice"
def INTRA_ELECTRODE_SEAL(name):
    return "intracellular_ephys/" + name + "/seal"
def INTRA_ELECTRODE_INIT_ACCESS_RESISTANCE(name):
    return "intracellular_ephys/" + name + "/initial_access_resistance"

def IMAGE_SITE_DESCRIPTION(site):
    return "optophysiology/" + site + "/description"
def IMAGE_SITE_MANIFOLD(site):
    return "optophysiology/" + site + "/manifold"
def IMAGE_SITE_INDICATOR(site):
    return "optophysiology/" + site + "/indicator"
def IMAGE_SITE_EXCITATION_LAMBDA(site):
    return "optophysiology/" + site + "/excitation_lambda"
def IMAGE_SITE_CHANNEL_1_NAME(site):
    return "optophysiology/" + site + "/channel_1_name"
def IMAGE_SITE_CHANNEL_1_LAMBDA(site):
    return "optophysiology/" + site + "/channel_1_emission_lambda"
def IMAGE_SITE_CHANNEL_2_NAME(site):
    return "optophysiology/" + site + "/channel_2_name"
def IMAGE_SITE_CHANNEL_2_LAMBDA(site):
    return "optophysiology/" + site + "/channel_2_emission_lambda"
def IMAGE_SITE_IMAGING_RATE(site):
    return "optophysiology/" + site + "/imaging_rate"
def IMAGE_SITE_LOCATION(site):
    return "optophysiology/" + site + "/location"
def IMAGE_SITE_DEVICE(site):
    return "optophysiology/" + site + "/device"
def IMAGE_SITE_CUSTOM(site, name):
    return "optophysiology/" + site + "/" + name

def OPTOGEN_SITE_DESCRIPTION(site):
    return "optogenetics/" + site + "/description"
def OPTOGEN_SITE_DEVICE(site):
    return "optogenetics/" + site + "/device"
def OPTOGEN_SITE_LAMBDA(site):
    return "optogenetics/" + site + "/lambda"
def OPTOGEN_SITE_LOCATION(site):
    return "optogenetics/" + site + "/location"
def OPTOGEN_SITE_CUSTOM(site, name):
    return "optogenetics/" + site + "/" + name

def DEVICE(name):
    return "devices/" + name



