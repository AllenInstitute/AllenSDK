from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from allensdk.api.api import Api
Api.default_api_url = 'http://testwarehouse:9000'

# tell the cache class what resolution (in microns) of data you want to download
mcc = MouseConnectivityCache(resolution=25)

# use the ontology class to get the id of the isocortex structure
ontology = mcc.get_ontology('ontology.csv')
isocortex = ontology['Isocortex']

# a list of dictionaries containing metadata for non-Cre experiments
experiments = mcc.get_experiments(file_name='non_cre.json',
                                  injection_structure_ids=isocortex['id'])

# download the projection density volume for one of the experiments
pd = mcc.get_projection_density(experiments[0]['id'])

