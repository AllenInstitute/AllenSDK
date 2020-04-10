import os

import pynwb

namespace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "AIBS_ecephys_namespace.yaml"))
pynwb.load_namespaces(namespace_path)


def to_dict(self) -> dict:
    out: dict = {}
    for key in self.__nwbfields__:
        if key not in ("name", "help"):
            out[key] = getattr(self, key)
    return out


EcephysProbe = pynwb.get_class('EcephysProbe', 'AIBS_ecephys')
EcephysLabMetaData = pynwb.get_class('EcephysLabMetaData', 'AIBS_ecephys')
EcephysLabMetaData.to_dict = to_dict
