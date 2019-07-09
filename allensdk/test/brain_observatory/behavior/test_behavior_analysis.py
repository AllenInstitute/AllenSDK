import pytest

from allensdk.brain_observatory.behavior.behavior_ophys_analysis import assign_to_interval
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession


@pytest.mark.requires_bamboo
def test_assign_to_interval():

    ophys_experiment_id = 789359614
    session = BehaviorOphysSession.from_lims(ophys_experiment_id)

    assignment = assign_to_interval(session.licks['time'].values, session.trials)
    for idx, lt in assignment.iteritems():
        assert lt in session.trials['lick_times'].loc[idx]
