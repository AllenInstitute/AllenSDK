from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

mca = MouseConnectivityApi()

# a list of dictionaries containing metadata for non-Cre experiments
experiments = mca.get_experiments(cre=False)

# download the projection density volume for one of the experiments
pd = mca.download_projection_density_volume(experiments[0]['id'], 'example.nrrd')

