import warnings

from allensdk.brain_observatory.behavior.behavior_ophys_experiment import \
        BehaviorOphysExperiment


class BehaviorOphysSession(BehaviorOphysExperiment):
    def __init__(self, **kwargs):
        warnings.warn(
            "allensdk.brain_observatory.behavior.behavior_ophys_session."
            "BehaviorOphysSession is deprecated. use "
            "allensdk.brain_observatory.behavior.behavior_ophys_experiment."
            "BehaviorOphysExperiment.",
            DeprecationWarning,
            stacklevel=3)
        super().__init__(**kwargs)
