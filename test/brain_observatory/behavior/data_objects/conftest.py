import pynwb
import pytest


@pytest.fixture
def data_object_roundtrip_fixture(tmp_path):
    def f(nwbfile, data_object_cls, **data_object_cls_kwargs):
        tmp_dir = tmp_path / "data_object_nwb_roundtrip_tests"
        tmp_dir.mkdir()
        nwb_path = tmp_dir / "data_object_roundtrip_nwbfile.nwb"

        with pynwb.NWBHDF5IO(str(nwb_path), 'w') as write_io:
            write_io.write(nwbfile)

        with pynwb.NWBHDF5IO(str(nwb_path), 'r') as read_io:
            roundtripped_nwbfile = read_io.read()

            data_object_instance = data_object_cls.from_nwb(
                roundtripped_nwbfile, **data_object_cls_kwargs
            )

        return data_object_instance

    return f
