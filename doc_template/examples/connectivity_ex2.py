from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

# tell the cache class what resolution (in microns) of data you want to download
mcc = MouseConnectivityCache(resolution=25)

# use the ontology class to get the id of the isocortex structure
ontology = mcc.get_ontology()
isocortex = ontology.get_structure_by_acronym('Isocortex')

# a list of dictionaries containing metadata for non-Cre experiments
experiments = mcc.get_experiments(injection_structure_ids=isocortex['id'])

# download the projection density volume for one of the experiments
pd = mcc.get_projection_density(experiments[0]['id'])

