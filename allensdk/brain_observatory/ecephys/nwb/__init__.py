import os

import pynwb
import numpy as np


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
        "lfp_sampling_rate",
        "has_lfp_data",
    )

    @pynwb.docval(
        {'name': 'name', 'type': str, 'doc': 'the name of this electrode'},
        {'name': 'description', 'type': str, 'doc': 'description of this electrode group'},
        {'name': 'location', 'type': str, 'doc': 'description of location of this electrode group'},
        {'name': 'device', 'type': pynwb.device.Device, 'doc': 'the device that was used to record from this electrode group'},
        {"name": "sampling_rate", "type": float, "doc": ""},
        {"name": "lfp_sampling_rate", "type": float, "doc": ""},
        {"name": "has_lfp_data", "type": (bool, np.bool_), "doc": ""},
        {'name': 'parent', 'type': 'NWBContainer', 'doc': 'The parent NWBContainer for this NWBContainer', 'default': None})
    def __init__(self, **kwargs):
        sampling_rate, lfp_sampling_rate, has_lfp_data = pynwb.popargs("sampling_rate", "lfp_sampling_rate", "has_lfp_data", kwargs)
        pynwb.call_docval_func(super(EcephysProbe, self).__init__, kwargs)
        self.sampling_rate, self.lfp_sampling_rate, self.has_lfp_data = sampling_rate, lfp_sampling_rate, has_lfp_data


@pynwb.register_class("EcephysLabMetaData", "AIBS_ecephys")
class EcephysLabMetaData(pynwb.file.LabMetaData):
    __nwbfields__ = (
        "name",
        "specimen_name",
        "age_in_days",
        "full_genotype",
        "strain",
        "sex",
        "stimulus_name"
    )

    @pynwb.docval(
        {'name': 'name', 'type': str, 'doc': 'name'},
        {'name': 'specimen_name', 'type': str, 'doc': 'full name of this specimen'},
        {'name': 'age_in_days', 'type': float, 'doc': 'age of this subject, in days'},
        {'name': 'full_genotype', 'type': str, 'doc': "long-form description of this subject's genotype"},
        {'name': 'strain', 'type': str, 'doc': "this subject's strain"},
        {'name': 'sex', 'type': str, 'doc': "this subject's sex"},
        {'name': 'stimulus_name', 'type': str, 'doc': "the name of the stimulus set used for this session"},
    )
    def __init__(self, **kwargs):

        (
            specimen_name, 
            age_in_days, 
            full_genotype, 
            strain, 
            sex, 
            stimulus_name 
        ) = pynwb.popargs(
            "specimen_name", 
            "age_in_days", 
            "full_genotype", 
            "strain", 
            "sex", 
            "stimulus_name",
        kwargs)

        pynwb.call_docval_func(super(EcephysLabMetaData, self).__init__, kwargs)

        (
            self.specimen_name, 
            self.age_in_days, 
            self.full_genotype, 
            self.strain, 
            self.sex, 
            self.stimulus_name 
        ) = (
            specimen_name, 
            age_in_days, 
            full_genotype, 
            strain, 
            sex, 
            stimulus_name
        )

    def to_dict(self):
        out = {}
        for key in self.__nwbfields__:
            if key not in ("name", "help"):
                out[key] = getattr(self, key)
        return out