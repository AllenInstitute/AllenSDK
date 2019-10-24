from functools import partial
import re
import os

import pytest
import pandas as pd

from allensdk.api import caching_utilities as cu


def get_data():
    return pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["duck", "kangaroo", "walrus", "ibex"]}
    )


def swapped_data():
    return pd.DataFrame(
        {"b": [1, 2, 3, 4], "a": ["duck", "kangaroo", "walrus", "ibex"]}
    )


def write_to_dict(dc, data):
    dc["data"] = data


def read_from_dict(dc):
    return dc["data"]


class InitiallyFailing:
    def __init__(self, succeed_at):
        self.succeed_at = succeed_at
        self.count = 0

    def __call__(self, fn):
        self.count += 1

        if self.count >= self.succeed_at:
            return fn()
        else:
            raise ValueError("foo!")


class InitiallyFailingWriter(InitiallyFailing):
    def __call__(self, dc, data):
        super(InitiallyFailingWriter, self).__call__(partial(write_to_dict, dc, data))


class InitiallyFailingReader(InitiallyFailing):
    def __call__(self, dc):
        return super(InitiallyFailingReader, self).__call__(partial(read_from_dict, dc))


class CallCountingCleanup:
    def __init__(self):
        self.count = 0

    def __call__(self, dc):
        self.count += 1
        dc.pop("data", None)


class CallCountingFetch:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return get_data()


def swap(data):
    data = data.copy()
    tmp = data["a"]
    data["a"] = data["b"]
    data["b"] = tmp
    return data


@pytest.mark.parametrize(
    "existing,fetch,write,read,pre_write,cleanup,lazy,num_tries,failure_message,expected,expected_fetches,expected_cleanups",
    [
        pytest.param(
            False,
            CallCountingFetch(),
            write_to_dict,
            InitiallyFailingReader(2),
            swap,
            CallCountingCleanup(),
            True,
            1,
            "",
            swapped_data(),
            1,
            0,
            id="simple case"
        ),
        pytest.param(
            False,
            CallCountingFetch(),
            write_to_dict,
            read_from_dict,
            swap,
            CallCountingCleanup(),
            False,
            0,
            "",
            swapped_data(),
            1,
            0,
            id="eager success case"
        ),
        pytest.param(
            False,
            CallCountingFetch(),
            write_to_dict,
            InitiallyFailingReader(3),
            swap,
            CallCountingCleanup(),
            True,
            1,
            "",
            "raise",
            1,
            1,
            id="lazy failure case"
        ),
        pytest.param(
            False,
            CallCountingFetch(),
            InitiallyFailingWriter(10),
            read_from_dict,
            swap,
            CallCountingCleanup(),
            True,
            12,
            "",
            swapped_data(),
            10,
            9,
            id="repeated failure case"
        ),
        pytest.param(
            False,
            CallCountingFetch(),
            InitiallyFailingWriter(10),
            read_from_dict,
            swap,
            CallCountingCleanup(),
            True,
            12,
            "bad news",
            "warn",
            10,
            9,
            id="warning case"
        ),
        pytest.param(
            False,
            CallCountingFetch(),
            write_to_dict,
            InitiallyFailingReader(2),
            swap,
            CallCountingCleanup(),
            False,
            1,
            "",
            "raise",
            1,
            1,
            id="eager failure case"
        ),
        pytest.param(
            True,
            CallCountingFetch(),
            write_to_dict,
            read_from_dict,
            None,
            CallCountingCleanup(),
            True,
            1,
            "",
            get_data(),
            0,
            0,
            id="existing data case"
        ),
    ],
)
def test_call_caching(
    existing,
    fetch,
    write,
    read,
    pre_write,
    cleanup,
    lazy,
    num_tries,
    failure_message,
    expected,
    expected_fetches,
    expected_cleanups,
):

    dc = {}
    if existing:
        write(dc, fetch())
        write.count = 0
        fetch.count = 0

    write_fn = partial(write, dc)
    read_fn = partial(read, dc)
    cleanup_fn = partial(cleanup, dc)

    fn = partial(
        cu.call_caching,
        fetch,
        write_fn,
        read_fn,
        pre_write,
        cleanup_fn,
        lazy,
        num_tries,
        failure_message
    )

    if isinstance(expected, str) and expected == "raise":
        with pytest.raises(ValueError):
            fn()
        assert not ("data" in dc)
    elif isinstance(expected, str) and expected == "warn":
        with pytest.warns(UserWarning) as warning:
            fn()
            assert re.match(f".*{failure_message}.*", str(warning.pop().message)) is not None
    else:
        pd.testing.assert_frame_equal(expected, fn(), check_like=True, check_dtype=False)

    assert expected_fetches == fetch.count
    assert expected_cleanups == cleanup.count


@pytest.mark.parametrize("existing", [True, False])
def test_one_file_call_caching(tmpdir_factory, existing):
    tmpdir = str(tmpdir_factory.mktemp("foo"))
    path = os.path.join(tmpdir, "baz.csv")

    getter = get_data
    data = getter()

    if existing:
        data.to_csv(path, index=False)
        getter = lambda: "foo"

    obtained = cu.one_file_call_caching(
        path,
        getter,
        lambda path, df: df.to_csv(path, index=False),
        lambda path: pd.read_csv(path),
        num_tries=2
    )

    pd.testing.assert_frame_equal(get_data(), obtained, check_like=True, check_dtype=False)
