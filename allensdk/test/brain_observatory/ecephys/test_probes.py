import pytest
import copy

from allensdk.brain_observatory.ecephys._channel import Channel
from allensdk.brain_observatory.ecephys.probes import Probes
from allensdk.brain_observatory.ecephys.write_nwb.schemas import Probe


@pytest.fixture(scope='module')
def probes_config_fixture(
        behavior_ecephys_session_config_fixture):
    """
    Return config data for a Probes object
    """

    input_data = copy.deepcopy(
        behavior_ecephys_session_config_fixture)

    # trim down the number of probes to reduce memory footprint of test
    input_data['probes'] = (
            input_data['probes'][:3])

    input_data = input_data['probes']
    return input_data


@pytest.fixture(scope='module')
def probes_fixture(
        probes_config_fixture):
    """
    Return a Probes object
    """
    probes = Probe().load(probes_config_fixture, many=True)

    # Don't test lfp here
    for probe in probes:
        probe['lfp'] = None
    obj = Probes.from_json(probes=probes)
    return obj


@pytest.mark.requires_bamboo
@pytest.mark.parametrize('roundtrip', [True, False])
def test_read_write_nwb(roundtrip,
                        data_object_roundtrip_fixture,
                        probes_fixture,
                        helper_functions):
    nwbfile = helper_functions.create_blank_nwb_file()
    probes_fixture.to_nwb(nwbfile=nwbfile)

    if roundtrip:
        obt = data_object_roundtrip_fixture(
            nwbfile=nwbfile,
            data_object_cls=Probes)
    else:
        obt = Probes.from_nwb(nwbfile=nwbfile)

    assert obt == probes_fixture


@pytest.mark.requires_bamboo
def test_skip_probes(
        probes_config_fixture):
    """tests that when skip_probes is passed that the probe is skipped"""
    names = [p['name'] for p in probes_config_fixture]
    skip_probes = [names[0]]
    probes = Probes.from_json(
        probes=probes_config_fixture, skip_probes=skip_probes)
    assert sorted([p.name for p in probes]) == \
           sorted([p for p in names if p not in skip_probes])


@pytest.mark.requires_bamboo
def test_units_from_structure_with_acronym(
        probes_fixture):
    """Checks that if there are channels with subregion in manual
    structure id, that units detected from this region are still included
    in units table"""
    expected_n_units = probes_fixture.get_units_table().shape[0]

    # Set the _structure_acronym to something with a hyphen
    for probe in probes_fixture.probes:
        for channel in probe.channels.value:
            if channel._structure_acronym == 'MGd':
                channel._structure_acronym = 'MGd-foo'
    obtained_n_units = probes_fixture.get_units_table().shape[0]

    assert expected_n_units == obtained_n_units


@pytest.mark.parametrize('structure_acronym', ('LGd-sh', 'LGd', None))
@pytest.mark.parametrize('strip_structure_subregion', (True, False))
def test_probe_channels_strip_subregion(
        structure_acronym, strip_structure_subregion):
    """Tests that subregion is stripped from manual structure acronym"""
    c = Channel(
        id=1,
        probe_channel_number=1,
        probe_vertical_position=1,
        probe_horizontal_position=1,
        probe_id=1,
        valid_data=True,
        structure_acronym=structure_acronym,
        strip_structure_subregion=strip_structure_subregion
    )
    if type(structure_acronym) is str:
        if strip_structure_subregion:
            expected = 'LGd'
        else:
            expected = 'LGd-sh' if structure_acronym == 'LGd-sh' \
                else 'LGd'
        assert c.structure_acronym == expected
    else:
        c.structure_acronym is None
