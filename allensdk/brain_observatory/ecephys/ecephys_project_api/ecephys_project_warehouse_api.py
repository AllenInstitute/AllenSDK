from allensdk.api.queries.rma_api import RmaApi
from .ecephys_project_api import EcephysProjectApi


class EcephysProjectWarehouseApi(EcephysProjectApi, RmaApi):
    pass