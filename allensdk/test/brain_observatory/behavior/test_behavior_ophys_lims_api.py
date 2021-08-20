from pathlib import Path

import pytest
import pandas as pd
import numpy as np
import h5py
import os
from contextlib import contextmanager

from allensdk.internal.api import OneResultExpectedError
from allensdk.brain_observatory.behavior.session_apis.data_io import (
    BehaviorOphysLimsApi, BehaviorOphysLimsExtractor)
from allensdk.brain_observatory.behavior.mtrain import ExtendedTrialSchema
from marshmallow.schema import ValidationError


@contextmanager
def does_not_raise(enter_result=None):
    """
    Context to help parametrize tests that may raise errors.
    If we start supporting only python 3.7+, switch to
    contextlib.nullcontext
    """
    yield enter_result


@pytest.mark.requires_bamboo
@pytest.mark.parametrize("ophys_experiment_id", [789359614])
def test_get_extended_trials(ophys_experiment_id):

    api = BehaviorOphysLimsApi(ophys_experiment_id)
    df = api.get_extended_trials()
    ets = ExtendedTrialSchema(partial=False, many=True)
    data_list_cs = df.to_dict("records")
    data_list_cs_sc = ets.dump(data_list_cs)
    ets.load(data_list_cs_sc)

    df_fail = df.drop(["behavior_session_uuid"], axis=1)
    ets = ExtendedTrialSchema(partial=False, many=True)
    data_list_cs = df_fail.to_dict("records")
    data_list_cs_sc = ets.dump(data_list_cs)
    try:
        ets.load(data_list_cs_sc)
        raise RuntimeError("This should have failed with "
                           "marshmallow.schema.ValidationError")
    except ValidationError:
        pass
