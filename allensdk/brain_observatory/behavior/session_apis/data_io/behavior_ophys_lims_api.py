from allensdk.brain_observatory.behavior.session_apis.data_transforms import \
    BehaviorOphysDataTransforms
from allensdk.core.cache_method_utilities import CachedInstanceMethodMixin


class BehaviorOphysLimsApi(BehaviorOphysDataTransforms,
                           CachedInstanceMethodMixin):
    pass
