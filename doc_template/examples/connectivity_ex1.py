from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi

mca = MouseConnectivityApi()

# get metadata for all non-Cre experiments
experiments = mca.experiment_source_search(injection_structures='root', transgenic_lines=0)

# download the projection density volume for one of the experiments
pd = mca.download_projection_density('example.nrrd', experiments[0]['id'], resolution=25)

