import os

import pynwb


namespace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "AIBS_ecephys_namespace.yaml"))
pynwb.load_namespaces(namespace_path)


@pynwb.register_class("EcephysProbe", "AIBS_ecephys")
class EcephysProbe(pynwb.ecephys.ElectrodeGroup):
    __nwbfields__ = (
        "name",
        "description",
        "location",
        "device",
        "sampling_rate",
        "lfp_sampling_rate"
    )

    @pynwb.docval(
        {'name': 'name', 'type': str, 'doc': 'the name of this electrode'},
        {'name': 'description', 'type': str, 'doc': 'description of this electrode group'},
        {'name': 'location', 'type': str, 'doc': 'description of location of this electrode group'},
        {'name': 'device', 'type': pynwb.device.Device, 'doc': 'the device that was used to record from this electrode group'},
        {"name": "sampling_rate", "type": float, "doc": ""},
        {"name": "lfp_sampling_rate", "type": float, "doc": ""},
        {'name': 'parent', 'type': 'NWBContainer', 'doc': 'The parent NWBContainer for this NWBContainer', 'default': None})
    def __init__(self, **kwargs):
        sampling_rate, lfp_sampling_rate = pynwb.popargs("sampling_rate", "lfp_sampling_rate", kwargs)
        pynwb.call_docval_func(super(EcephysProbe, self).__init__, kwargs)
        self.sampling_rate, self.lfp_sampling_rate = sampling_rate, lfp_sampling_rate