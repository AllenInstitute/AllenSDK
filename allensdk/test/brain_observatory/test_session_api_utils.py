import warnings
from inspect import Parameter

import pytest
import numpy as np
import pandas as pd

from allensdk.brain_observatory.session_api_utils import is_equal, ParamsMixin


class ParamsMixinTestHarness(ParamsMixin):

    def __init__(self, param_to_ignore, a_param_1: int, a_param_2: float,
                 b_param_1: list, c_param_1: bool, d_param_1: np.ndarray,
                 e_param_1: pd.Series, f_param_1: pd.DataFrame):

        super().__init__(ignore={'param_to_ignore'})

        self._a_param_1 = a_param_1
        self._a_param_2 = a_param_2
        self._b_param_1 = b_param_1
        self._c_param_1 = c_param_1
        self._d_param_1 = d_param_1
        self._e_param_1 = e_param_1
        self._f_param_1 = f_param_1


@pytest.fixture
def mixin_harness_fixture(request) -> ParamsMixinTestHarness:
    param_to_ignore = request.param.get('param_to_ignore', 'x')
    a_param_1 = request.param.get('a_param_1', 8)
    a_param_2 = request.param.get('a_param_2', 42.0)
    b_param_1 = request.param.get('b_param_1', [1, 2, 3])
    c_param_1 = request.param.get('c_param_1', True)
    d_param_1 = request.param.get('d_param_1', np.array([5, 5]))
    e_param_1 = request.param.get('e_param_1', pd.Series([4.0, 5.0]))
    f_param_1 = request.param.get('f_param_1', pd.DataFrame([1, 2, 3]))

    mixed_in = ParamsMixinTestHarness(param_to_ignore, a_param_1,
                                      a_param_2, b_param_1, c_param_1,
                                      d_param_1, e_param_1, f_param_1)
    mixed_in._updated_params = request.param.get('updated_params', set())

    return mixed_in


@pytest.mark.parametrize("a, b, expected", [
    (2, 2, True),
    ('1', '1', True),
    (1.5, 1.5, True),
    ([1, 2, 3], [1, 2, 3], True),
    ({1, 2, 3}, {1, 2, 3}, True),
    ({'a', 'b', 'c'}, {'c', 'a', 'b'}, True),
    ({'a': 0, 'z': 42}, {'a': 0, 'z': 42}, True),
    (np.array([1, 2, 3]), np.array([1, 2, 3]), True),
    ({'c': np.array([5, 5])}, {'c': np.array([5, 5])}, True),
    (pd.Series([5, 5, 5]), pd.Series([5, 5, 5]), True),
    (pd.DataFrame([10, 10]), pd.DataFrame([10, 10]), True),
    ([pd.DataFrame(['a', 'b', 'c'])], [pd.DataFrame(['a', 'b', 'c'])], True),
    ({'a': np.array([1, 2, 3])}, {'a': np.array([1, 2, 3])}, True),
    ({'a': {'x': pd.Series([5.0, 6.0])}}, {'a': {'x': pd.Series([5.0, 6.0])}}, True),
    ({'a': 20, 'b': 30}, {'b': 30, 'a': 20}, True),

    (1, 2.0, False),
    ('1', 2, False),
    ([1, 2, 3], 5, False),
    ([1, 2, 3], [1, 2], False),
    ([1, 2, 3], [3, 2, 1], False),
    (['a', 'b'], {'a', 'b'}, False),
    ({'a'}, {'a', 'b'}, False),
    ({'a'}, {'b'}, False),
    ({'a', 'b'}, np.array(['a', 'b']), False),
    (np.array([3, 4, 5]), np.array([3, 4]), False),
    ({'c': np.array([5, 5])}, {'c': np.array([5, 6])}, False),
    (pd.Series([5, 5, 5]), pd.Series([5, 6, 5]), False),
    (pd.Series([1, 2, 3]), pd.Series([1, 2]), False),
    (pd.DataFrame([10, 10]), pd.DataFrame([10, 7]), False),
    (pd.DataFrame([10, 20, 30]), pd.DataFrame([10, 20]), False),
    ([pd.DataFrame(['a', 'b', 'c'])], [pd.DataFrame(['a', 'b', 'd'])], False),
    ({'a': np.array([1, 2, 3])}, {'a': np.array([1, 2, 5])}, False),
    ({'a': {'x': pd.Series([5.0, 6.0])}}, {'a': {'x': pd.Series([5.0, 7.0])}}, False),

    (pd.Series([5, 5, 5]), np.array([5, 5, 5]), False),
    (np.array([8, 8, 8]), pd.DataFrame([8, 8, 8]), False),
    (pd.Series([3, 3, 3]), pd.DataFrame([3, 3, 3]), False),
])
def test_is_equal(a, b, expected):
    assert is_equal(a, b) == expected


@pytest.mark.parametrize("mixin_harness_fixture, expected", [
    ({},
     [Parameter('param_to_ignore', Parameter.POSITIONAL_OR_KEYWORD),
      Parameter('a_param_1', Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
      Parameter('a_param_2', Parameter.POSITIONAL_OR_KEYWORD, annotation=float),
      Parameter('b_param_1', Parameter.POSITIONAL_OR_KEYWORD, annotation=list),
      Parameter('c_param_1', Parameter.POSITIONAL_OR_KEYWORD, annotation=bool),
      Parameter('d_param_1', Parameter.POSITIONAL_OR_KEYWORD, annotation=np.ndarray),
      Parameter('e_param_1', Parameter.POSITIONAL_OR_KEYWORD, annotation=pd.Series),
      Parameter('f_param_1', Parameter.POSITIONAL_OR_KEYWORD, annotation=pd.DataFrame)]),
], indirect=["mixin_harness_fixture"])
def test_get_param_signatures(mixin_harness_fixture, expected):
    obtained = mixin_harness_fixture._get_param_signatures()
    assert obtained == expected


@pytest.mark.parametrize("mixin_harness_fixture, expected", [
    ({},
     {'param_to_ignore': Parameter.empty, 'a_param_1': int,
      'a_param_2': float, 'b_param_1': list, 'c_param_1': bool,
      'd_param_1': np.ndarray, 'e_param_1': pd.Series,
      'f_param_1': pd.DataFrame}),
], indirect=["mixin_harness_fixture"])
def test_get_param_type_annotations(mixin_harness_fixture, expected):
    obtained = mixin_harness_fixture._get_param_type_annotations()
    assert obtained == expected


@pytest.mark.parametrize("mixin_harness_fixture, expected", [
    ({},
     ['a_param_1', 'a_param_2', 'b_param_1', 'c_param_1', 'd_param_1',
      'e_param_1', 'f_param_1', 'param_to_ignore']),
], indirect=["mixin_harness_fixture"])
def test_get_param_names(mixin_harness_fixture, expected):
    obtained = mixin_harness_fixture._get_param_names()
    assert obtained == expected


@pytest.mark.parametrize("mixin_harness_fixture, expected", [
    ({},
     {'a_param_1': 8, 'a_param_2': 42.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({'a_param_1': 2, 'a_param_2': 10.0, 'b_param_1': [1], 'c_param_1': False},
     {'a_param_1': 2, 'a_param_2': 10.0, 'b_param_1': [1],
      'c_param_1': False, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])})
], indirect=["mixin_harness_fixture"])
def test_get_params(mixin_harness_fixture, expected):
    obtained = mixin_harness_fixture.get_params()
    is_equal(obtained, expected)


@pytest.mark.parametrize("mixin_harness_fixture, params_to_set, expected", [
    ({},
     {'a_param_1': 5},
     {'a_param_1': 5, 'a_param_2': 42.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({},
     {'a_param_2': 10.0},
     {'a_param_1': 8, 'a_param_2': 10.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({},
     {'b_param_1': [3, 4, 5]},
     {'a_param_1': 8, 'a_param_2': 42.0, 'b_param_1': [3, 4, 5],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({},
     {'a_param_1': 20, 'a_param_2': 3.14, 'b_param_1': [9, 10]},
     {'a_param_1': 20, 'a_param_2': 3.14, 'b_param_1': [9, 10],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({},
     {'d_param_1': np.array([20, 20]), 'e_param_1': pd.Series([1, 2, 3]), 'b_param_1': [9, 10]},
     {'a_param_1': 20, 'a_param_2': 3.14, 'b_param_1': [9, 10],
      'c_param_1': True, 'd_param_1': np.array([20, 20]),
      'e_param_1': pd.Series([1, 2, 3]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

], indirect=["mixin_harness_fixture"])
def test_set_params_basic(mixin_harness_fixture, params_to_set, expected):
    mixin_harness_fixture.set_params(**params_to_set)
    obtained = mixin_harness_fixture.get_params()
    is_equal(obtained, expected)


@pytest.mark.parametrize("mixin_harness_fixture, params_to_set, expected", [
    ({},
     {'a_param': 5},
     {'a_param_1': 8, 'a_param_2': 42.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({},
     {'something_random': 10.0},
     {'a_param_1': 8, 'a_param_2': 42.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

], indirect=["mixin_harness_fixture"])
def test_set_params_with_invalid_params(mixin_harness_fixture,
                                        params_to_set, expected):
    with warnings.catch_warnings(record=True) as w:
        mixin_harness_fixture.set_params(**params_to_set)
        assert 'not valid and is being ignored' in str(w[-1].message)

    obtained = mixin_harness_fixture.get_params()
    is_equal(obtained, expected)


@pytest.mark.parametrize("mixin_harness_fixture, params_to_set, expected", [
    ({},
     {'a_param_1': 'hello'},
     {'a_param_1': 8, 'a_param_2': 42.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({},
     {'a_param_2': [5]},
     {'a_param_1': 8, 'a_param_2': 42.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({},
     {'b_param_1': 1},
     {'a_param_1': 8, 'a_param_2': 42.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({},
     {'d_param_1': [1, 2, 3]},
     {'a_param_1': 8, 'a_param_2': 42.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

    ({},
     {'e_param_1': {1, 2, 3}},
     {'a_param_1': 8, 'a_param_2': 42.0, 'b_param_1': [1, 2, 3],
      'c_param_1': True, 'd_param_1': np.array([5, 5]),
      'e_param_1': pd.Series([4.0, 5.0]), 'f_param_1': pd.DataFrame([1, 2, 3])}),

], indirect=["mixin_harness_fixture"])
def test_set_params_with_invalid_type(mixin_harness_fixture, params_to_set, expected):
    with warnings.catch_warnings(record=True) as w:
        mixin_harness_fixture.set_params(**params_to_set)
        assert 'should be of type' in str(w[-1].message)

    obtained = mixin_harness_fixture.get_params()
    is_equal(obtained, expected)


@pytest.mark.parametrize("mixin_harness_fixture, params_to_set, data_params, expected", [
    ({},
     {'a_param_1': 42},
     {'a_param_1'},
     True),

    ({},
     {'a_param_1': 8},
     {'a_param_1'},
     False),

    ({},
     {'a_param_1': 8.0},
     {'a_param_1'},
     False),

    ({},
     {'a_param_2': 3.0},
     {'a_param_1'},
     False),

    ({},
     {'a_param_2': 2.5, 'b_param_1': ['a', 'b', 'c']},
     {'b_param_1'},
     True),

    ({},
     {'a_param_1': 10, 'a_param_2': 9.0},
     {'b_param_1'},
     False),

    ({},
     {'d_param_1': np.array([98, 99, 100]), 'a_param_2': 9.0},
     {'d_param_1'},
     True),

    ({},
     {'d_param_1': np.array([98, 99, 100]), 'e_param_1': pd.Series([1, 3, 5])},
     {'e_param_1'},
     True),

    ({},
     {'d_param_1': np.array([98, 99, 100]), 'e_param_1': pd.Series([4.0, 5.0])},
     {'e_param_1'},
     False),


], indirect=["mixin_harness_fixture"])
def test_needs_data_refresh(mixin_harness_fixture, params_to_set, data_params, expected):
    mixin_harness_fixture.set_params(**params_to_set)
    obtained = mixin_harness_fixture.needs_data_refresh(data_params)
    assert obtained == expected


@pytest.mark.parametrize("mixin_harness_fixture, data_params, expected", [
    ({'updated_params': {'a_param_1', 'b_param_1'}},
     {'a_param_1'},
     {'b_param_1'}),

    ({'updated_params': {'a_param_1', 'a_param_2', 'b_param_1'}},
     {'a_param_1', 'a_param_2'},
     {'b_param_1'}),
], indirect=["mixin_harness_fixture"])
def test_clear_updated_params(mixin_harness_fixture, data_params, expected):
    mixin_harness_fixture.clear_updated_params(data_params)
    assert mixin_harness_fixture._updated_params == expected
