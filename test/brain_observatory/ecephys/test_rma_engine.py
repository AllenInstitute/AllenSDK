import pytest
import pandas as pd
import numpy as np

import allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine as rma_engine


@pytest.mark.parametrize("dataframe,expected_types", [
    [
        pd.DataFrame({
            "a": ["1", "2", "3"],
            "b": ["a", "1", "2"]
        }),
        {"a": np.dtype("int64"), "b": np.dtype("O")}
    ],
    [
        pd.DataFrame({
            "a": ["1", "2.4", "3"],
            "b": ["a", "1", "2"]
        }),
        {"a": float, "b": np.dtype("O")}
    ]
])
def test_infer_column_types(dataframe, expected_types):

    obtained = rma_engine.infer_column_types(dataframe)
    obtained_types = {colname: obtained[colname].dtype for colname in obtained.columns}
    
    assert(set(expected_types.keys()) == set(obtained_types.keys()))

    for key, value in expected_types.items():
        assert np.dtype(value) == np.dtype(obtained_types[key])
