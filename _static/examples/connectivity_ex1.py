from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
import nrrd

mca = MouseConnectivityApi()

# get metadata for all non-Cre experiments
experiments = mca.experiment_source_search(injection_structures='root', transgenic_lines=0)

# download the projection density volume for one of the experiments
mca.download_projection_density('example.nrrd', experiments[0]['id'], resolution=25)

# read it into memory
pd_array, pd_info = nrrd.read('example.nrrd')

